"""Real dataset labels remain visible when a housing cannot enter training."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.ui.dialogs import TrainingDialog
from test_smoke_training import make_service


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def capture(service, index, label="GOOD", part_id="LEMON", session=""):
    image = np.random.default_rng(index).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    return service.capture_sample(image, "inner_face", label, part_id, session)


def test_imported_good_images_stay_good_when_shared_housing_has_ng(app, tmp_path, monkeypatch):
    service = make_service(tmp_path)
    samples = [capture(service, 1), capture(service, 2), capture(service, 3, "NG")]
    originals = {Path(sample["path"]): Path(sample["path"]).read_bytes() for sample in samples}
    metadata = {path.with_suffix(path.suffix + ".json"): path.with_suffix(path.suffix + ".json").read_bytes()
                for path in originals}
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(str(args[-1])))
    dialog = TrainingDialog(service)

    statistics = service.dataset_stats()
    assert statistics["good_images"] == 2
    assert statistics["ng_images"] == 1
    assert statistics["good_parts"] == 0
    assert statistics["ng_parts"] == 1
    assert "LEMON" in dialog.readiness.text()
    assert "NG" in dialog.readiness.text()
    assert "GOOD" in dialog.readiness.text()
    assert warnings == []

    dialog.start_job()
    deadline = time.monotonic() + 5
    while dialog.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.003)
    assert not dialog.busy
    assert len(warnings) == 1
    assert "LEMON" in warnings[0]
    assert "3 independent GOOD" in warnings[0]
    assert [sample["label"] for sample in service.samples()].count(0) == 2
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert all(path.read_bytes() == original for path, original in metadata.items())
    dialog.close()


def test_saved_session_grouping_is_used_in_training_summary(app, tmp_path):
    service = make_service(tmp_path)
    service.config.setdefault("dataset", {})["group_sessions"] = True
    for index in range(3):
        capture(service, index, part_id=f"actual-housing-{index}", session="shared-session")
    dialog = TrainingDialog(service)
    readiness = service.dataset_stats()["training_readiness"]
    assert readiness["parts"] == 3
    assert readiness["eligible_good_groups"] == 1
    assert dialog.group_sessions.isChecked()
    assert "session" in dialog.readiness.text().lower()
    dialog.close()
