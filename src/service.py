"""Application use cases shared by the desktop HMI and command-line workflows."""
from __future__ import annotations
from collections import Counter, deque
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import threading
import time
import uuid
from typing import Callable

import cv2
import numpy as np

from src.acquisition import CameraConfig, CameraService, QualityChecker, runtime_diagnostics
from src.artifacts import ArtifactStore, atomic_json, digest, safe_name
from src.security import TechnicianLock
from src.data import DatasetIntegrityError, DatasetStore, Sample, split_by_part
from src.evaluation import calibrate_operating_point, evaluate_samples, export_report
from src.inspection.evidence import EvidenceStore
from src.inspection.session import InspectionSession, ViewDecision
from src.inspection.view_pipeline import ViewPipeline
from src.utils.config import deep_merge
from src.utils.filesystem import display_path, filesystem_path
from src.utils.image_utils import read_image, write_image
from src.vision import FaceIdentifier, GeometryProfile, HousingLocator, LandmarkRegistrar, roi_mask, tile_image


MODES = {'geometry', 'efficientad', 'hybrid'}


def _jsonable(value):
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class NeuroIrisService:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.dataset = DatasetStore(config['paths']['dataset'])
        self.artifacts = ArtifactStore(config['paths']['artifacts'], config['product'])
        self.evidence = EvidenceStore(config['paths']['results'])
        self.camera: CameraService | None = None
        self.session: InspectionSession | None = None
        self.frames: dict[str, np.ndarray] = {}
        self.marked: dict[str, np.ndarray] = {}
        self._pipelines: dict[tuple, ViewPipeline] = {}
        self._recent = deque(maxlen=12)
        self._recent_lock = threading.Lock()
        self._diagnostics = None
        self._last_data = None
        self._operation_lock = threading.RLock()
        self._technician = TechnicianLock(Path(config['paths']['runtime']) / 'technician-lock.json')
        self._saved_event_id = None
        self._capture_after = time.monotonic()
        self._inspection_mode = 'automatic'

    @property
    def inspection_mode(self):
        return self._inspection_mode

    def set_inspection_mode(self, mode):
        if mode not in {'automatic', 'manual'}:
            raise ValueError('Inspection mode must be automatic or manual')
        with self._operation_lock:
            self._require_idle()
            self._inspection_mode = mode

    @property
    def technician_unlocked(self):
        return self._technician.unlocked

    def technician_configured(self):
        return self._technician.configured

    def set_technician_pin(self, pin):
        self._technician.setup(pin)

    def unlock_technician(self, pin):
        return self._technician.unlock(pin)

    def lock_technician(self):
        self._technician.lock()

    def diagnostics(self) -> dict:
        if self._diagnostics is None:
            self._diagnostics = runtime_diagnostics()
        return {**self._diagnostics, 'camera': self.camera.diagnostics() if self.camera else {'status': 'disconnected'},
                'selected_backend': self.config['models']['backend'],
                'model_store': display_path(self.artifacts.root),
                'loaded_models': [{'view': key[0], 'mode': key[1], 'version': pipeline.metadata['version'],
                                   'inference_device': str(getattr(pipeline.model, 'device', 'cpu')) if pipeline.model else 'cpu'}
                                  for key, pipeline in self._pipelines.items()],
                'optional_models': {'dinomaly': 'Not integrated or benchmarked; unavailable',
                                    'dinov2': 'Disabled; requires a separately validated adapter'},
                'metrology': 'Image pixels/normalized geometry; no validated physical calibration'}

    def dataset_stats(self) -> dict:
        return self.dataset.stats(group_sessions=bool(self.config.get('dataset', {}).get('group_sessions', False)))

    def samples(self, view=None) -> list[dict]:
        # A pending registration must not prevent opening Dataset Manager to fix it.
        return [sample.to_dict() for sample in self.dataset.inventory(view, strict=False)]

    def pending_samples(self, view=None) -> list[dict]:
        return self.dataset.unmanaged_images(view)

    def import_sample(self, path, view, label, part_id, session_id='') -> dict:
        return self.dataset.import_image(path, view, label, part_id, session_id).to_dict()

    def repair_sample(self, path, view, label, part_id, session_id='') -> dict:
        return self.dataset.repair_image(path, view, label, part_id, session_id).to_dict()

    def capture_sample(self, image, view, label, part_id, session_id='') -> dict:
        return self.dataset.capture_image(image, view, label, part_id, session_id).to_dict()

    def remove_sample(self, sample_id):
        self.dataset.remove(sample_id)

    def _roi_path(self, view):
        return filesystem_path(self.config['paths']['runtime']) / 'rois' / safe_name(self.config['product']) / (safe_name(view) + '.json')

    def get_rois(self, view) -> list[dict]:
        path = self._roi_path(view)
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

    def save_rois(self, view, definitions):
        self._technician.require()
        self._require_idle()
        roi_mask((1000, 1000, 3), definitions)  # validate before replacing saved configuration
        atomic_json(self._roi_path(view), definitions)
        self._pipelines.clear()

    def _require_idle(self):
        if self.session is not None:
            if self.session.final_decision is None:
                raise RuntimeError('Finalize or cancel the current part before changing settings')
            if self._saved_event_id != self.session.event_id:
                raise RuntimeError('Evidence was not saved. Retry finalizing the current part before continuing.')

    def configure(self, settings: dict):
        self._technician.require()
        if not isinstance(settings, dict):
            raise TypeError('Settings must be a dictionary')
        self._require_idle()
        candidate = deep_merge(self.config, settings)
        if candidate['models']['backend'] not in MODES:
            raise ValueError('Select geometry, efficientad or hybrid; optional neural backends are not installed')
        QualityChecker(candidate.get('quality'))
        required = candidate['views']['required']
        if not required or len(set(required)) != len(required) or not set(required) <= set(candidate['views']['enabled']):
            raise ValueError('Required views must be nonempty, unique enabled views')
        geometry = candidate['geometry']
        if not 0 < geometry.get('rib_inner_radius', .33) < geometry.get('rib_outer_radius', .96) <= 1:
            raise ValueError('Rib annulus ratios must satisfy 0 < inner < outer <= 1')
        for name in ('rib_position_tolerance_px', 'rib_contrast_tolerance', 'rib_mad_multiplier', 'minimum_rib_gap_radius', 'rim_tolerance_px', 'mounting_edge_tolerance_px'):
            value = geometry.get(name, 1.0)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        cal = candidate['calibration']
        if not 0 <= cal['target_fpr'] <= cal['recheck_fpr'] < 1:
            raise ValueError('Calibration targets require 0 <= target_fpr <= recheck_fpr < 1')
        tile = candidate['tiling']
        if int(tile['tile_size']) < 8 or not 0 <= float(tile['overlap']) < 1 or int(tile['batch_size']) < 1:
            raise ValueError('Invalid tile size, overlap or batch size')
        path = filesystem_path(self.config['paths']['runtime']) / 'config-overrides.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        atomic_json(path, _jsonable(deep_merge(previous, settings)))
        self.config = candidate
        self._pipelines.clear()

    def set_mode(self, backend):
        self.configure({'models': {'backend': backend}})

    def start_camera(self, index=0, width=3840, height=2160, fps=30, backend='auto'):
        self.stop_camera()
        self.camera = CameraService(CameraConfig(index=int(index), width=int(width), height=int(height), fps=float(fps), backend=backend))
        self.camera.start()

    def stop_camera(self):
        if self.camera:
            self.camera.stop()
        with self._recent_lock:
            self._recent.clear()
            self._capture_after = time.monotonic()

    def set_camera_control(self, name, value):
        self._technician.require()
        self._require_idle()
        if not self.camera:
            raise RuntimeError('Connect the selected camera first')
        return self.camera.set_control(name, value)

    def preview(self):
        if not self.camera:
            return None
        packet = self.camera.latest()
        if packet is None:
            return None
        with self._recent_lock:
            if not self._recent or self._recent[-1].sequence != packet.sequence:
                self._recent.append(packet)
        return {'image': packet.image, 'sequence': packet.sequence,
                'timestamp': packet.timestamp, 'diagnostics': self.camera.diagnostics()}

    def begin_part(self, part_id):
        with self._operation_lock:
            if self.session is not None and self.session.final_decision is None:
                raise RuntimeError('Finish or cancel the current physical part first')
            self._require_idle()
            self.session = InspectionSession(part_id, tuple(self.config['views']['required']),
                                             inspection_mode=self.inspection_mode)
            self._saved_event_id = None
            self._capture_after = time.monotonic()
            with self._recent_lock:
                self._recent.clear()
            self.frames, self.marked = {}, {}
            return self._session_dict()

    def _session_dict(self):
        if self.session is None:
            return {}
        data = self.session.to_dict()
        aggregate = self.session.preview().to_dict()
        return {**data, 'verdict': aggregate['verdict'], 'status': self.session.state.value,
                'reasons': aggregate['reasons']}

    def _validate_model_configuration(self, metadata, view):
        settings = metadata.get('settings')
        if not isinstance(settings, dict):
            raise ValueError('Saved model has no calibrated inspection settings. Select a complete model or retrain this view.')
        if any(settings.get(section, {}) != _jsonable(self.config.get(section, {}))
               for section in ('quality', 'registration', 'geometry', 'calibration', 'tiling', 'localization')):
            raise RuntimeError('Inspection settings changed after calibration. Retrain this view or restore the settings saved with its model.')
        if metadata.get('rois', []) != self.get_rois(view):
            raise RuntimeError('The ROI configuration changed after calibration. Retrain this view before inspection.')

    def model_readiness(self, views=None):
        """Report saved-model availability without loading neural weights.

        This is a setup check for the dashboard. Inference still verifies every
        model file and loads the actual pipeline before producing a verdict.
        No mode, selected version or inspection configuration is changed here.
        """
        backend = self.config['models']['backend']
        requested = list(self.config['views']['required'] if views is None else views)
        if not set(requested) <= set(self.config['views']['enabled']):
            raise ValueError('Model readiness requires enabled housing views')
        names = {'inner_face': 'Face A', 'ribbed_face': 'Face B',
                 'side_left': 'Left side', 'side_right': 'Right side'}

        def saved(view, mode):
            directory = self.artifacts.active(view, mode, verify_files=False)
            metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            required = ['template.png', 'geometry.npz', 'thresholds.json']
            if mode in {'hybrid', 'efficientad'}:
                required += ['neural/efficientad.pt', 'neural/model.json']
            listed = {Path(name).as_posix() for name in metadata['files']}
            for name in required:
                if name not in listed or not (directory / name).is_file():
                    raise ValueError(f'Saved model is incomplete: {name}. Model folder: {display_path(directory)}')
            self._validate_model_configuration(metadata, view)
            return directory, metadata

        rows = []
        for view in requested:
            row = {'view': view, 'ready': False, 'status': 'missing', 'version': None,
                   'reason': '', 'available_modes': []}
            try:
                directory, metadata = saved(view, backend)
                row.update(ready=True, status='ready', version=metadata['version'],
                           path=display_path(directory))
            except FileNotFoundError as exc:
                row['reason'] = str(exc)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                row.update(status='invalid', reason=str(exc))
            if not row['ready']:
                for mode in sorted(MODES - {backend}):
                    try:
                        saved(view, mode)
                        row['available_modes'].append(mode)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        continue
                try:
                    last_training = self.training_screening(view)
                    if last_training.get('status') == 'failed' and last_training.get('error'):
                        row['last_training_error'] = str(last_training['error'])
                        row['reason'] += f' Last training did not complete: {last_training["error"]}'
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    pass  # A damaged audit report cannot conceal model availability.
            rows.append(row)
        ready = bool(rows) and all(row['ready'] for row in rows)
        details = [f'{names.get(row["view"], row["view"])}: '
                   + (f'saved model {row["version"]}' if row['ready'] else f'{row["status"]} {backend} model')
                   + (f' (available: {", ".join(row["available_modes"])})' if row['available_modes'] else '')
                   for row in rows]
        legacy = any((self.artifacts.root.parent / name).is_file()
                     for name in ('patchcore/model.joblib', 'visual_changenet/model.joblib'))
        message = f'{backend.capitalize()} models — ' + '; '.join(details)
        if not ready:
            message += '. Open Train Model for the missing face, or Model Settings to select a trained mode/version.'
            message += f' Model folder: {display_path(self.artifacts.root)}'
            if legacy:
                message += ' First-version model files were found; they require training in the new view-specific model format.'
        return {'backend': backend, 'ready': ready, 'artifact_root': display_path(self.artifacts.root),
                'views': rows, 'message': message, 'legacy_models_found': legacy}

    def _pipeline(self, view, backend):
        directory = self.artifacts.active(view, backend)
        key = (view, backend, directory.name)
        if key not in self._pipelines:
            metadata = ArtifactStore.verify(directory)
            self._validate_model_configuration(metadata, view)
            self._pipelines[key] = ViewPipeline(directory, backend)
        return self._pipelines[key]

    def _infer(self, image, view, backend=None):
        backend = backend or self.config['models']['backend']
        if view == 'auto':
            view, confidence = FaceIdentifier().identify(image)
            if view not in {'inner_face', 'ribbed_face'} or confidence < .65:
                return ViewDecision('unidentified', 'RECHECK', ['Face identity is uncertain; select the housing face manually']), image.copy(), None
        if view not in self.config['views']['enabled']:
            raise ValueError(f'Unknown or disabled view: {view}')
        try:
            pipeline = self._pipeline(view, backend)
            return self._infer_with_pipeline(image, view, pipeline)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            return ViewDecision(view, 'RECHECK', [str(exc)], metrics={'mode': backend, 'error': type(exc).__name__}), image.copy(), None

    def _infer_with_pipeline(self, image, view, pipeline):
        if pipeline.view != view:
            raise ValueError('Model bundle view does not match the capture view')
        return pipeline.inspect(image)

    def inspect_frame(self, image, view, part_id, *, capture_type='imported still image; temporal motion not assessed', capture_context=None):
        with self._operation_lock:
            if self.inspection_mode != 'automatic':
                raise RuntimeError('Select automatic inspection mode before running model inspection')
            if self.session is None:
                self.begin_part(part_id)
            if self.session.part_id != str(part_id).strip():
                raise ValueError('The image must belong to the active physical part ID')
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.dtype != np.uint8 or image.shape[2] != 3:
                raise ValueError('Inspection needs a native uint8 BGR image')
            decision, marked, data = self._infer(image, view)
            decision.metrics['capture_type'] = capture_type
            if capture_context is not None:
                decision.metrics['capture_context'] = _jsonable(capture_context)
            if decision.view == 'unidentified':
                self.session.record_attempt(decision, 'unidentified_capture')
                return {**decision.to_dict(), 'marked_image': marked, 'session': self._session_dict()}
            self.session.record(decision)
            self.frames[decision.view] = image.copy()
            self.marked[decision.view] = marked.copy()
            self._last_data = data
            return {**decision.to_dict(), 'marked_image': marked, 'session': self._session_dict()}

    def record_manual_decision(self, view, verdict, part_id, image=None):
        """Record an operator assessment with its actual native image, without inference."""
        with self._operation_lock:
            if self.inspection_mode != 'manual':
                raise RuntimeError('Select manual inspection mode before recording an operator decision')
            if view not in self.config['views']['enabled']:
                raise ValueError('Select an explicit enabled housing face for manual inspection')
            if verdict not in {'PASS', 'FAIL', 'RECHECK'}:
                raise ValueError('Verdict must be PASS, FAIL or RECHECK')
            if (not isinstance(image, np.ndarray) or image.ndim != 3 or
                    image.dtype != np.uint8 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
                raise ValueError('Capture or open a native color image before recording a decision')
            if self.session is None:
                self.begin_part(part_id)
            if self.session.part_id != str(part_id).strip():
                raise ValueError('The image must belong to the active physical part ID')
            decision = ViewDecision(view, verdict, ['View inspection complete'], metrics={
                'decision_source': 'operator', 'inspection_scope': 'manual',
                'display_decision': {'PASS': 'OK', 'FAIL': 'NG', 'RECHECK': 'View Recheck'}[verdict],
                'capture_type': 'manual inspection image; automatic capture quality not assessed',
            })
            self.session.record(decision)
            self.frames[view] = image.copy()
            self.marked[view] = image.copy()
            self._last_data = None
            return {**decision.to_dict(), 'marked_image': image.copy(), 'session': self._session_dict()}

    def inspect_live(self, view, part_id):
        """Explicit camera provenance: stale stills can never bypass live gates."""
        with self._operation_lock:
            if self.inspection_mode != 'automatic':
                raise RuntimeError('Select automatic inspection mode before running model inspection')
            if self.session is None:
                self.begin_part(part_id)
            if self.session.part_id != str(part_id).strip():
                raise ValueError('Physical part ID differs from the active session')
            packet = self.camera.latest() if self.camera else None
            diagnostic = self.camera.diagnostics() if self.camera else {}
            if packet is None or not diagnostic.get('connected'):
                raise RuntimeError('Camera is disconnected or has no current frame; reconnect before inspection')
            age = time.monotonic() - packet.timestamp
            maximum_age = QualityChecker(self.config['quality']).settings['max_frame_age']
            if not np.isfinite(age) or not 0 <= age <= maximum_age or packet.timestamp < self._capture_after:
                decision = ViewDecision(view, 'RECHECK', ['Current camera frame is stale; wait for a fresh stable capture'], metrics={'capture_type': 'live acquisition gate'})
                self.session.record_attempt(decision)
                return {**decision.to_dict(), 'marked_image': packet.image.copy(), 'session': self._session_dict()}
            if diagnostic.get('resolution_matches_request') is False or diagnostic.get('format_verified') is False:
                decision = ViewDecision(view, 'RECHECK', ['Camera did not negotiate the requested native resolution/MJPEG format. Reconnect or commission the actual capture format.'], metrics={'capture_type': 'live acquisition gate', 'camera': diagnostic})
                self.session.record_attempt(decision)
                return {**decision.to_dict(), 'marked_image': packet.image.copy(), 'session': self._session_dict()}
            with self._recent_lock:
                if not self._recent or self._recent[-1].sequence != packet.sequence:
                    self._recent.append(packet)
                recent = [p for p in self._recent if p.timestamp >= self._capture_after]
            pose = HousingLocator().detect(packet.image)
            selected = QualityChecker(self.config['quality']).select_sharpest(recent, mask=pose.mask if pose else None) if pose else None
            if selected is None:
                self.session.record_attempt(ViewDecision(view, 'RECHECK', ['Wait for a stationary, focused, complete housing in fresh camera frames'], metrics={'capture_type': 'live acquisition gate'}))
                return {'view': view, 'verdict': 'RECHECK',
                        'reasons': ['Wait for a stationary, focused, complete housing in fresh camera frames'],
                        'candidates': [], 'metrics': {}, 'marked_image': packet.image.copy(),
                        'session': self._session_dict()}
            # Shared still pipeline; only acquisition gating differs from file evaluation.
            result = self.inspect_frame(selected.image, view, part_id, capture_type='live stable-frame selection', capture_context={'sequence': selected.sequence, 'timestamp_monotonic_seconds': selected.timestamp, 'camera': diagnostic})
            self._capture_after = time.monotonic()
            with self._recent_lock:
                self._recent.clear()
            return result

    def finish_part(self):
        with self._operation_lock:
            if self.session is None:
                raise RuntimeError('Start a part inspection first')
            decision = self.session.finalize()
            directory = self.evidence.save(self.session, self.frames, self.marked)
            self._saved_event_id = self.session.event_id
            return {**decision.to_dict(), 'session': self._session_dict(), 'evidence_path': display_path(directory)}

    def cancel_part(self):
        with self._operation_lock:
            if self.session is None:
                raise RuntimeError('No active part to cancel')
            decision = self.session.cancel('Cancelled by operator; inspection coverage incomplete')
            directory = self.evidence.save(self.session, self.frames, self.marked)
            self._saved_event_id = self.session.event_id
            return {**decision.to_dict(), 'session': self._session_dict(), 'evidence_path': display_path(directory)}

    def history(self):
        return [{**row, 'verdict': (row.get('final_decision') or {}).get('verdict', 'RECHECK'),
                 'timestamp': row.get('completed_at') or row.get('created_at', '')}
                for row in self.evidence.history()]

    def models(self):
        return self.artifacts.list()

    def select_model(self, view, version, backend):
        self._technician.require()
        self._require_idle()
        self.artifacts.select(view, version, backend)
        self._pipelines.clear()

    def diagnostic_maps(self):
        data = self._last_data
        if data is None:
            return {}
        return {'anomaly_map': data.anomaly_map,
                'registered_image': data.registration.image if data.registration is not None else None,
                'timings': data.timings,
                'geometry': data.geometry.measurements if data.geometry is not None else {}}

    def _settings_snapshot(self):
        return _jsonable({key: self.config[key] for key in
                          ('registration', 'geometry', 'quality', 'models', 'tiling', 'localization', 'calibration')})

    @staticmethod
    def _provisional_manifest(directory, metadata):
        files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json'}
        atomic_json(directory / 'manifest.json', {**metadata, 'schema_version': 1, 'files': files})

    def _screening_directory(self, view):
        return (filesystem_path(self.config['paths']['results']) / 'training_screening' /
                safe_name(self.config['product']) / safe_name(view))

    def training_screening(self, view):
        """Latest read-only screening report, including failed training attempts."""
        path = self._screening_directory(view) / 'latest.json'
        if not path.is_file():
            return {}
        report = json.loads(path.read_text(encoding='utf-8'))
        # An external audit write can fail after atomic model activation. The
        # completed, immutable bundle then remains the authoritative run record.
        if report.get('status') != 'completed' and report.get('backend') in MODES:
            try:
                directory = self.artifacts.root / safe_name(view)
                pointer = json.loads((directory / f'active-{report["backend"]}.json').read_text(encoding='utf-8'))
                bundled_path = directory / safe_name(pointer['version']) / 'training_screening.json'
                bundled = json.loads(bundled_path.read_text(encoding='utf-8'))
                if bundled.get('run_id') == report.get('run_id') and bundled.get('status') == 'completed':
                    return {**bundled, 'report_path': display_path(bundled_path)}
            except (OSError, KeyError, TypeError, ValueError):
                pass
        return report

    def _save_training_screening(self, report):
        # Keep audit records outside model staging so failed attempts retain them.
        payload = _jsonable(report)
        atomic_json(Path(report['report_path']), payload)
        atomic_json(self._screening_directory(report['view']) / 'latest.json', payload)

    def _screen_training_images(self, samples, settings, rois, report, emit):
        """Exclude unsuitable GOOD train captures without modifying the dataset."""
        quality = QualityChecker(settings['quality'])
        locator = HousingLocator()
        report.update(candidate_images=len(samples), candidate_sample_ids=[s.id for s in samples],
                      quality_settings=quality.settings, registration_settings=settings['registration'],
                      rois=copy.deepcopy(rois))
        self._save_training_screening(report)
        excluded_ids = set()
        screening_percent = 2

        def notify(percent, message):
            nonlocal screening_percent
            screening_percent = max(screening_percent, percent)
            emit(screening_percent, 'Training image screening', message)

        def exclude(sample, stage, reasons, metrics):
            excluded_ids.add(sample.id)
            report['excluded'].append({'sample_id': sample.id, 'filename': sample.path.name,
                                       'path': display_path(sample.path), 'part_id': sample.part_id,
                                       'view': sample.view, 'sha256': sample.sha256,
                                       'stage': stage, 'reasons': list(reasons), 'metrics': metrics})
            report['excluded_images'] = len(report['excluded'])
            self._save_training_screening(report)
            notify(screening_percent,
                   f'Automatically excluded {sample.path.name}: {"; ".join(reasons)}. Original retained.')

        def load(sample):
            # Integrity violations remain fatal; they are not capture-quality failures.
            if digest(sample.path) != sample.sha256:
                raise DatasetIntegrityError(f'Training image bytes changed after inventory: {sample.path}')
            return read_image(sample.path)

        candidates = []
        for index, sample in enumerate(samples):
            notify(2 + int(5 * index / max(1, len(samples))),
                   f'Checking capture quality {index + 1}/{len(samples)}: {sample.path.name}')
            try:
                image = load(sample)
            except (ValueError, cv2.error) as exc:
                if isinstance(exc, DatasetIntegrityError):
                    raise
                exclude(sample, 'quality', ['unreadable_image', str(exc)], {})
                continue
            pose = locator.detect(image)
            checked = quality.check(image, mask=pose.mask if pose else None)
            reasons = list(checked.reasons)
            if pose is None:
                reasons.append('complete_housing_not_located')
            elif not pose.complete:
                reasons.append('housing_clipped_at_frame_edge')
            if reasons:
                exclude(sample, 'quality', reasons, checked.metrics)
                continue
            candidates.append((sample, checked.metrics, [image.shape[1], image.shape[0]]))
            # Do not retain a second full-resolution copy of every candidate.
            del image

        def registered_mask(registration, metrics):
            confidence = float(registration.confidence)
            metrics = {**metrics, 'registration_confidence': confidence if np.isfinite(confidence) else None}
            if (not registration.valid or not np.isfinite(registration.confidence) or
                    registration.confidence < settings['registration']['min_confidence']):
                reasons = list(registration.reasons) if not registration.valid else []
                return None, 'registration', reasons or ['insufficient_registration_confidence'], metrics
            pose = locator.detect(registration.image)
            if pose is None or not pose.complete:
                return None, 'aligned_pose', ['registered_housing_not_complete'], metrics
            mask = cv2.bitwise_and(pose.mask, roi_mask(registration.image.shape, rois))
            pixels = int(np.count_nonzero(mask))
            metrics['roi_pixels'] = pixels
            minimum_pixels = max(30, int(np.ceil(.02 * np.pi * pose.radius ** 2)))
            metrics['minimum_roi_pixels'] = minimum_pixels
            if pixels < minimum_pixels:
                return None, 'roi', ['inspection_roi_has_too_few_part_pixels'], metrics
            return mask, None, [], metrics

        template = registrar = template_registration = None
        template_sample = None
        notify(8, 'Checking reference-image landmarks and inspection ROI')
        # A sharp silhouette alone is not a usable master: require internal landmarks.
        # Prefer the represented native resolution over a lone sharp thumbnail.
        resolution_counts = Counter(tuple(row[2]) for row in candidates)
        report['template_selection'] = 'Most represented native resolution, then highest sharpness; valid landmarks and ROI required'
        for sample, metrics, _ in sorted(candidates, key=lambda row: (
                -resolution_counts[tuple(row[2])], -row[1].get('sharpness', 0))):
            image = load(sample)
            try:
                candidate_registrar = LandmarkRegistrar(image, settings['registration'])
                registration = candidate_registrar.register(image)
                mask, stage, reasons, measured = registered_mask(registration, metrics)
            except cv2.error as exc:
                exclude(sample, 'registration', ['registration_error', str(exc)], metrics)
                continue
            if mask is None:
                exclude(sample, stage, reasons, measured)
                continue
            template, registrar, template_registration = image, candidate_registrar, registration
            template_sample = sample
            report.update(template_sample_id=sample.id, template_sha256=sample.sha256)
            break

        accepted, aligned, masks, resolutions = [], [], [], []
        if template is not None:
            for sample, metrics, resolution in candidates:
                if sample.id in excluded_ids:
                    continue
                try:
                    registration = (template_registration if sample.id == template_sample.id else
                                    registrar.register(load(sample)))
                    mask, stage, reasons, measured = registered_mask(registration, metrics)
                except cv2.error as exc:
                    exclude(sample, 'registration', ['registration_error', str(exc)], metrics)
                    continue
                if mask is None:
                    exclude(sample, stage, reasons, measured)
                    continue
                accepted.append(sample)
                aligned.append(registration.image)
                masks.append(mask)
                resolutions.append(tuple(resolution))
                report['accepted_sample_ids'] = [s.id for s in accepted]
                report['accepted_images'] = len(accepted)
                notify(12, f'{len(accepted)} usable training captures; '
                           f'{report["excluded_images"]} automatically excluded')
        report.update(status='screened', accepted_images=len(accepted),
                      accepted_sample_ids=[s.id for s in accepted],
                      accepted_part_ids=sorted({s.part_id for s in accepted}),
                      source_resolutions=sorted(set(resolutions)))
        self._save_training_screening(report)
        if not accepted:
            reasons = sorted({reason for row in report['excluded'] for reason in row['reasons']})
            raise ValueError(
                f'No usable GOOD training captures remain: 0 accepted; {report["excluded_images"]} of '
                f'{report["candidate_images"]} automatically excluded ({"; ".join(reasons)}). '
                'Originals were retained. Recapture clear, fully visible housings with identifiable '
                f'landmarks, and review the inspection ROI. Report: {report["report_path"]}')
        notify(14, f'Using {len(accepted)} GOOD captures; {report["excluded_images"]} automatically excluded. '
                   f'Report: {report["report_path"]}')
        return template, accepted, aligned, masks

    def _screen_validation_images(self, samples, pipeline, ng_part_ids, screening, emit, *, phase):
        """Apply live capture gates without fitting to or reassigning validation parts."""
        validation, invalid = [], []
        for index, sample in enumerate(samples):
            emit(20 if phase == 'preflight' else 75, 'Independent validation ' + phase,
                 f'Checking reserved validation capture {index + 1}/{len(samples)}: {sample.path.name}')
            data = pipeline.analyze(read_image(sample.path))
            if data.valid:
                validation.append((sample, data))
                continue
            metrics = dict(data.quality.metrics) if data.quality is not None else {}
            if data.registration is not None:
                metrics['registration_confidence'] = data.registration.confidence
            invalid.append({'sample_id': sample.id, 'part_id': sample.part_id,
                            'filename': sample.path.name, 'view': sample.view,
                            'path': display_path(sample.path), 'label': 'NG' if sample.label else 'GOOD',
                            'stage': data.failure_stage or 'analysis', 'reasons': list(data.reasons),
                            'metrics': metrics})
        good_samples = [sample for sample in samples if sample.label == 0 and sample.part_id not in ng_part_ids]
        valid_good = [(sample, data) for sample, data in validation
                      if sample.label == 0 and sample.part_id not in ng_part_ids]
        screening['invalid_validation_captures'] = invalid
        screening['validation_screening'] = {
            'phase': phase, 'status': 'passed' if valid_good else 'failed',
            'candidate_images': len(samples), 'accepted_images': len(validation),
            'excluded_images': len(invalid), 'good_candidate_images': len(good_samples),
            'good_accepted_images': len(valid_good),
            'reserved_good_part_ids': sorted({sample.part_id for sample in good_samples}),
            'accepted_sample_ids': [sample.id for sample, _ in validation],
            'expected_source_resolutions': pipeline.metadata.get('source_resolutions', []),
        }
        self._save_training_screening(screening)
        if not valid_good:
            good_ids = {sample.id for sample in good_samples}
            blockers = Counter(reason for row in invalid if row['sample_id'] in good_ids
                               for reason in set(row['reasons']))
            reasons = '; '.join(f'{reason} ({count} capture(s))' for reason, count in blockers.most_common(3))
            parts = ', '.join(sorted({sample.part_id for sample in good_samples}))
            raise ValueError(
                'No independent GOOD validation captures passed quality and landmark registration; '
                f'calibration cannot proceed (0 of {len(good_samples)} accepted; reserved parts: {parts}). '
                f'Validation blockers: {reasons or "no eligible GOOD validation captures"}. '
                'Review the validation capture details and recapture these same physical parts at the '
                'training resolution with clear, fully visible housings and identifiable landmarks. '
                f'Report: {screening["report_path"]}')
        return validation, valid_good, invalid

    def train(self, view, backend='efficientad', progress: Callable | None = None):
        with self._operation_lock:
            self._require_idle()
            if backend not in MODES:
                raise ValueError('EfficientAD and geometry are the implemented training backends')
            if view not in self.config['views']['enabled']:
                raise ValueError('Select an enabled view')
            run_id = uuid.uuid4().hex
            screening = {'schema_version': 1, 'run_id': run_id, 'view': view,
                         'backend': 'efficientad' if backend == 'hybrid' else backend,
                         'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'screening',
                         'candidate_images': 0, 'accepted_images': 0, 'excluded_images': 0,
                         'candidate_sample_ids': [], 'accepted_sample_ids': [], 'excluded': [],
                         'report_path': display_path(self._screening_directory(view) / (run_id + '.json'))}
            self._save_training_screening(screening)
            try:
                result = self._train(view, backend, progress, screening)
            except Exception as exc:
                screening.update(status='failed', error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
                self._save_training_screening(screening)
                raise
            screening = copy.deepcopy(result['training_screening'])
            try:
                self._save_training_screening(screening)
            except OSError as exc:
                # Publication has succeeded. Preserve its successful outcome and
                # point the user at the already-written, verified bundle report.
                screening['report_path'] = display_path(Path(result['artifact_path']) / 'training_screening.json')
                result['warnings'].append(
                    f'Model saved successfully; the external screening report could not be finalized: {exc}. '
                    f'The completed report is retained in the model bundle: {screening["report_path"]}')
            return {**result, 'training_screening': screening}

    def _train(self, view, backend, progress, screening):
        if backend not in MODES:
            raise ValueError('EfficientAD and geometry are the implemented training backends')
        if view not in self.config['views']['enabled']:
            raise ValueError('Select an enabled view')
        backend = 'efficientad' if backend == 'hybrid' else backend
        emit = progress or (lambda *_: None)
        emit(0, 'Dataset integrity', 'Checking physical IDs, hashes and global view grouping')
        pending = self.pending_samples()
        if pending:
            names = ', '.join(Path(item['path']).name for item in pending[:3])
            raise DatasetIntegrityError(
                f'{len(pending)} dataset image(s) need registration or metadata repair ({names}). '
                'Open Dataset Manager, select the rows marked Needs Registration or Needs Repair, '
                'enter their actual physical part ID, view, label and session, then click '
                'Register / Repair Selected. Process only one physical housing in each batch. '
                'Your images have been kept; do not delete them or invent IDs from filenames.'
            )
        samples = self.dataset.inventory()
        split = split_by_part(samples, seed=int(self.config['runtime']['random_seed']),
                              ratios=tuple(self.config.get('dataset', {}).get('split', [.7, .15, .15])),
                              group_sessions=bool(self.config.get('dataset', {}).get('group_sessions', False)))
        selected = {name: [s for s in getattr(split, name) if s.view == view]
                    for name in ('train', 'validation', 'test')}
        good_train = [s for s in selected['train'] if s.label == 0]
        good_val = [s for s in selected['validation'] if s.label == 0]
        if not good_train or not good_val or not selected['test']:
            raise ValueError('This view needs independent GOOD train/validation parts and untouched test parts. Capture each configured face for the same physical IDs.')
        settings = self._settings_snapshot()
        rois = self.get_rois(view)
        started = time.perf_counter()
        template, good_train, aligned, masks = self._screen_training_images(
            good_train, settings, rois, screening, emit)
        emit(15, 'Geometry', f'Learning {view} geometry using GOOD training parts only')
        geometry = GeometryProfile(view, settings['geometry'])
        geometry.fit(aligned, masks=masks)
        version = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        metadata = {'version': version, 'product': self.config['product'], 'view': view,
                    'backend': backend, 'settings': settings, 'rois': rois,
                    'source_resolutions': screening['source_resolutions'],
                    'split': split.manifest(), 'training_parts': sorted({s.part_id for s in split.train}),
                    'training_accepted_sample_ids': list(screening['accepted_sample_ids']),
                    'validation_parts': sorted({s.part_id for s in split.validation}),
                    'test_parts': sorted({s.part_id for s in split.test}),
                    'packages': {name: importlib.metadata.version(name) for name in
                                 ('numpy', 'opencv-python', 'PyQt6', 'scikit-learn')},
                    'limitations': ['No real housing accuracy or camera validation is implied by software training',
                                    'GOOD-only calibration cannot establish defect recall' if not any(s.label for s in selected['validation']) else 'Pilot targets require measured held-out acceptance']}
        with self.artifacts.staging(view) as stage:
            write_image(stage / 'template.png', template)
            geometry.save(stage / 'geometry.npz')
            # Check all non-neural live gates before expensive network setup and
            # fitting. The split and training master remain fixed; validation
            # captures are never used to fit geometry or select a new template.
            provisional = {'geometry': {'reject_threshold': 1e30, 'recheck_threshold': 1e30},
                           'anomaly': {'reject_threshold': 1e30, 'recheck_threshold': 1e30}}
            atomic_json(stage / 'thresholds.json', provisional)
            self._provisional_manifest(stage, metadata)
            ng_part_ids = {s.part_id for s in samples if s.label == 1}
            preflight = ViewPipeline(stage, 'geometry')
            validation, valid_good, invalid_validation = self._screen_validation_images(
                selected['validation'], preflight, ng_part_ids, screening, emit, phase='preflight')
            if backend == 'efficientad':
                # Native validation analyses need not occupy memory during fit.
                validation.clear()
                valid_good.clear()
                del preflight
                from src.models.efficientad import EfficientADBackend
                tiles = [tile.image for image, mask in zip(aligned, masks)
                         for tile in tile_image(image, settings['tiling']['tile_size'], settings['tiling']['overlap'], mask)]
                maximum = int(settings['tiling'].get('max_training_tiles', 2000))
                if len(tiles) > maximum:
                    # Explicitly sample training tiles only; retain native source footprints.
                    rng = np.random.default_rng(self.config['runtime']['random_seed'])
                    tiles = [tiles[i] for i in sorted(rng.choice(len(tiles), maximum, replace=False))]
                model = EfficientADBackend(settings['models']['efficientad'])
                emit(25, 'EfficientAD', f'Training the real teacher/student/autoencoder on {len(tiles)} GOOD native tiles')
                neural = model.fit(tiles, stage / 'neural', progress=lambda percent, message: emit(min(72, 25 + int(percent * .47)), 'EfficientAD', message))
                metadata['neural'] = neural
                # Reuse exactly the same live capture gates with the fitted
                # neural branch to collect calibration scores.
                self._provisional_manifest(stage, metadata)
                pipeline = ViewPipeline(stage, 'hybrid')
                validation, valid_good, invalid_validation = self._screen_validation_images(
                    selected['validation'], pipeline, ng_part_ids, screening, emit, phase='calibration')
            emit(75, 'Independent calibration', 'Measuring raw scores on validation parts; test set remains untouched')
            branches = 2 if backend == 'efficientad' else 1
            budgets = branches * len(self.config['views']['required'])
            limits = {'part_fpr_target': settings['calibration']['target_fpr'],
                      'branch_view_budget_divisor': budgets,
                      'note': 'Conservative branch/view allocation; empirical targets are not population guarantees'}
            for branch in (['geometry', 'anomaly'] if backend == 'efficientad' else ['geometry']):
                value = (lambda data: data.geometry.score) if branch == 'geometry' else (lambda data: data.anomaly_score)
                good_scores = [float(value(d)) for s, d in valid_good]
                ng = [(s, d) for s, d in validation if s.label == 1]
                op = calibrate_operating_point(good_scores, good_part_ids=[s.part_id for s, _ in valid_good],
                                               ng_scores=[float(value(d)) for s, d in ng], ng_part_ids=[s.part_id for s, _ in ng],
                                               target_fpr=settings['calibration']['target_fpr'] / budgets,
                                               recheck_fpr=settings['calibration']['recheck_fpr'] / budgets,
                                               training_part_ids=metadata['training_parts'], test_part_ids=metadata['test_parts'],
                                               minimum_reject_score=1.0 if branch == 'geometry' else 0.0,
                                               minimum_recheck_score=.8 if branch == 'geometry' else 0.0)
                limits[branch] = op.to_dict()
            limits['invalid_validation_captures'] = invalid_validation
            atomic_json(stage / 'thresholds.json', limits)
            self._provisional_manifest(stage, metadata)
            pipeline = ViewPipeline(stage, 'hybrid' if backend == 'efficientad' else 'geometry')
            emit(90, 'Blind evaluation', 'Running the exact live pipeline on untouched test physical parts')
            report = evaluate_samples(selected['test'], lambda image, _: pipeline.inspect(image)[0],
                                      mandatory_views=[view], training_part_ids=metadata['training_parts'],
                                      validation_part_ids=metadata['validation_parts'],
                                      training_hashes=[s.sha256 for s in split.train],
                                      validation_hashes=[s.sha256 for s in split.validation],
                                      score_key='anomaly_score' if backend == 'efficientad' else 'geometry_score')
            report['scope'] = f'Single-view {view} evaluation; final parts require every configured mandatory view'
            report['invalid_validation_captures'] = invalid_validation
            export_report(report, stage / 'evaluation')
            summary = {'version': version, 'view': view, 'backend': backend,
                       'training_images': len(good_train), 'training_parts': len({s.part_id for s in good_train}),
                       'validation_parts': len({s.part_id for s in selected['validation']}),
                       'elapsed_seconds': time.perf_counter() - started, 'thresholds': limits,
                       'test': report['parts'], 'report': report,
                       'warnings': metadata['limitations'] + report.get('warnings', [])}
            completed_screening = {**screening, 'status': 'completed',
                                   'finished_at': datetime.now(timezone.utc).isoformat()}
            metadata['training_screening'] = completed_screening
            summary['training_screening'] = completed_screening
            if screening['excluded_images']:
                summary['warnings'].append(f'{screening["excluded_images"]} unsuitable training captures were '
                                           'automatically excluded; original files and labels were retained.')
            atomic_json(stage / 'training_screening.json', completed_screening)
            atomic_json(stage / 'training_summary.json', summary)
            target = self.artifacts.publish(stage, metadata)
            self._pipelines.clear()
            emit(100, 'Training completed', f'Persisted model bundle {target.name}; held-out report saved')
            return {**summary, 'artifact_path': display_path(target)}

    @staticmethod
    def _saved_samples(rows):
        result = []
        for row in rows:
            path = filesystem_path(row['path'])
            if not path.is_file() or digest(path) != row['sha256']:
                raise ValueError(f'Original held-out sample missing or modified: {path}')
            result.append(Sample(row['id'], path, int(row['label']), row['view'], row['part_id'], row['sha256'], row.get('metadata', {})))
        return result

    def evaluate(self, view, backend=None, progress=None):
        with self._operation_lock:
            backend = backend or self.config['models']['backend']
            if view == 'all':
                from src.benchmark import evaluate_all
                return evaluate_all(self, backend, progress)
            pipeline = self._pipeline(view, backend)
            meta = pipeline.metadata
            split = meta['split']
            samples = self._saved_samples([row for row in split['test'] if row['view'] == view])
            if progress:
                progress(0, 'Blind evaluation', 'Using the saved held-out partition; no fitting or threshold updates')
            report = evaluate_samples(samples, lambda image, _: pipeline.inspect(image)[0], mandatory_views=[view],
                                      training_part_ids=meta['training_parts'], validation_part_ids=meta['validation_parts'],
                                      training_hashes=[s['sha256'] for s in split['train']],
                                      validation_hashes=[s['sha256'] for s in split['validation']],
                                      score_key='geometry_score' if backend == 'geometry' else 'anomaly_score')
            report.update({'scope': f'Single-view {view}; final production sessions require all mandatory views',
                           'model_version': meta['version'], 'backend': backend})
            output = filesystem_path(self.config['paths']['results']) / 'evaluations' / (uuid.uuid4().hex)
            export_report(report, output)
            if progress:
                progress(100, 'Evaluation completed', f'Actual outcomes saved to {output}')
            return {**report, 'report_path': display_path(output)}

    def benchmark(self, view, progress=None):
        from src.benchmark import benchmark
        with self._operation_lock:
            return benchmark(self, view, progress)
