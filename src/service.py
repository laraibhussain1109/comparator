"""Application use cases shared by the desktop HMI and command-line workflows."""
from __future__ import annotations
from collections import deque
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
from src.data import DatasetStore, Sample, split_by_part
from src.evaluation import calibrate_operating_point, evaluate_samples, export_report
from src.inspection.evidence import EvidenceStore
from src.inspection.session import InspectionSession, ViewDecision
from src.inspection.view_pipeline import ViewPipeline
from src.utils.config import deep_merge
from src.utils.image_utils import read_image
from src.vision import FaceIdentifier, GeometryProfile, HousingLocator, LandmarkRegistrar, roi_mask, tile_image


MODES = {'geometry', 'efficientad', 'hybrid'}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
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
                'loaded_models': [{'view': key[0], 'mode': key[1], 'version': pipeline.metadata['version'],
                                   'inference_device': str(getattr(pipeline.model, 'device', 'cpu')) if pipeline.model else 'cpu'}
                                  for key, pipeline in self._pipelines.items()],
                'optional_models': {'dinomaly': 'Not integrated or benchmarked; unavailable',
                                    'dinov2': 'Disabled; requires a separately validated adapter'},
                'metrology': 'Image pixels/normalized geometry; no validated physical calibration'}

    def dataset_stats(self) -> dict:
        return self.dataset.stats()

    def samples(self, view=None) -> list[dict]:
        return [sample.to_dict() for sample in self.dataset.inventory(view)]

    def import_sample(self, path, view, label, part_id, session_id='') -> dict:
        return self.dataset.import_image(path, view, label, part_id, session_id).to_dict()

    def capture_sample(self, image, view, label, part_id, session_id='') -> dict:
        return self.dataset.capture_image(image, view, label, part_id, session_id).to_dict()

    def remove_sample(self, sample_id):
        self.dataset.remove(sample_id)

    def _roi_path(self, view):
        return Path(self.config['paths']['runtime']) / 'rois' / safe_name(self.config['product']) / (safe_name(view) + '.json')

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
        path = Path(self.config['paths']['runtime']) / 'config-overrides.json'
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
            self.session = InspectionSession(part_id, tuple(self.config['views']['required']))
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

    def _pipeline(self, view, backend):
        directory = self.artifacts.active(view, backend)
        key = (view, backend, directory.name)
        if key not in self._pipelines:
            metadata = ArtifactStore.verify(directory)
            if any(metadata['settings'].get(section, {}) != _jsonable(self.config.get(section, {}))
                   for section in ('quality', 'registration', 'geometry', 'calibration', 'tiling', 'localization')):
                raise RuntimeError('Inspection settings changed after calibration. Retrain this view or restore the settings saved with its model.')
            if metadata.get('rois', []) != self.get_rois(view):
                raise RuntimeError('The ROI configuration changed after calibration. Retrain this view before inspection.')
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

    def inspect_live(self, view, part_id):
        """Explicit camera provenance: stale stills can never bypass live gates."""
        with self._operation_lock:
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
            return {**decision.to_dict(), 'session': self._session_dict(), 'evidence_path': str(directory)}

    def cancel_part(self):
        with self._operation_lock:
            if self.session is None:
                raise RuntimeError('No active part to cancel')
            decision = self.session.cancel('Cancelled by operator; inspection coverage incomplete')
            directory = self.evidence.save(self.session, self.frames, self.marked)
            self._saved_event_id = self.session.event_id
            return {**decision.to_dict(), 'session': self._session_dict(), 'evidence_path': str(directory)}

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

    def train(self, view, backend='efficientad', progress: Callable | None = None):
        with self._operation_lock:
            self._require_idle()
            return self._train(view, backend, progress)

    def _train(self, view, backend, progress):
        if backend not in MODES:
            raise ValueError('EfficientAD and geometry are the implemented training backends')
        if view not in self.config['views']['enabled']:
            raise ValueError('Select an enabled view')
        backend = 'efficientad' if backend == 'hybrid' else backend
        emit = progress or (lambda *_: None)
        emit(0, 'Dataset integrity', 'Checking physical IDs, hashes and global view grouping')
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
        quality = QualityChecker(settings['quality'])
        locator = HousingLocator()
        rois = self.get_rois(view)
        started = time.perf_counter()
        decoded = []
        for sample in good_train:
            image = read_image(sample.path)
            pose = locator.detect(image)
            report = quality.check(image, mask=pose.mask if pose else None)
            if pose is None or not pose.complete or not report.valid:
                raise ValueError(f'Invalid GOOD training image {sample.path.name}: {report.reasons or ["complete housing not located"]}. Reject or recapture it in Dataset Manager.')
            decoded.append((image, report.metrics.get('sharpness', 0)))
        template = max(decoded, key=lambda item: item[1])[0]
        registrar = LandmarkRegistrar(template, settings['registration'])
        aligned = []
        masks = []
        for image, _ in decoded:
            registration = registrar.register(image)
            if not registration.valid or registration.confidence < settings['registration']['min_confidence']:
                raise ValueError('GOOD training image cannot be registered with internal landmarks: ' + '; '.join(registration.reasons))
            image = registration.image
            pose = locator.detect(image)
            aligned.append(image)
            masks.append(cv2.bitwise_and(pose.mask, roi_mask(image.shape, rois)))
        emit(15, 'Geometry', f'Learning {view} geometry using GOOD training parts only')
        geometry = GeometryProfile(view, settings['geometry'])
        geometry.fit(aligned, masks=masks)
        version = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        metadata = {'version': version, 'product': self.config['product'], 'view': view,
                    'backend': backend, 'settings': settings, 'rois': rois,
                    'source_resolutions': sorted({(image.shape[1], image.shape[0]) for image, _ in decoded}),
                    'split': split.manifest(), 'training_parts': sorted({s.part_id for s in split.train}),
                    'validation_parts': sorted({s.part_id for s in split.validation}),
                    'test_parts': sorted({s.part_id for s in split.test}),
                    'packages': {name: importlib.metadata.version(name) for name in
                                 ('numpy', 'opencv-python', 'PyQt6', 'scikit-learn')},
                    'limitations': ['No real housing accuracy or camera validation is implied by software training',
                                    'GOOD-only calibration cannot establish defect recall' if not any(s.label for s in selected['validation']) else 'Pilot targets require measured held-out acceptance']}
        with self.artifacts.staging(view) as stage:
            if not cv2.imwrite(str(stage / 'template.png'), template):
                raise OSError('Could not save registration template')
            geometry.save(stage / 'geometry.npz')
            if backend == 'efficientad':
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
            # Fixed provisional limits exist only to collect validation raw scores.
            provisional = {'geometry': {'reject_threshold': 1e30, 'recheck_threshold': 1e30},
                           'anomaly': {'reject_threshold': 1e30, 'recheck_threshold': 1e30}}
            atomic_json(stage / 'thresholds.json', provisional)
            self._provisional_manifest(stage, metadata)
            pipeline = ViewPipeline(stage, 'hybrid' if backend == 'efficientad' else 'geometry')
            emit(75, 'Independent calibration', 'Measuring raw scores on validation parts; test set remains untouched')
            validation = []
            invalid_validation = []
            for sample in selected['validation']:
                data = pipeline.analyze(read_image(sample.path))
                if data.valid:
                    validation.append((sample, data))
                else:
                    invalid_validation.append({'sample_id': sample.id, 'part_id': sample.part_id, 'reasons': data.reasons})
            ng_part_ids = {s.part_id for s in samples if s.label == 1}
            valid_good = [(s, d) for s, d in validation if s.label == 0 and s.part_id not in ng_part_ids]
            if not valid_good:
                raise ValueError('No independent GOOD validation captures passed quality and landmark registration; calibration cannot proceed')
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
            atomic_json(stage / 'training_summary.json', summary)
            target = self.artifacts.publish(stage, metadata)
            self._pipelines.clear()
            emit(100, 'Training completed', f'Persisted model bundle {target.name}; held-out report saved')
            return {**summary, 'artifact_path': str(target)}

    @staticmethod
    def _saved_samples(rows):
        result = []
        for row in rows:
            path = Path(row['path'])
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
            output = Path(self.config['paths']['results']) / 'evaluations' / (uuid.uuid4().hex)
            export_report(report, output)
            if progress:
                progress(100, 'Evaluation completed', f'Actual outcomes saved to {output}')
            return {**report, 'report_path': str(output)}

    def benchmark(self, view, progress=None):
        from src.benchmark import benchmark
        with self._operation_lock:
            return benchmark(self, view, progress)
