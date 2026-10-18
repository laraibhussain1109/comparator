from src.security import TechnicianLock
import pytest


def test_technician_pin_is_hashed_and_restart_locks_settings(tmp_path):
    path = tmp_path / 'technician.json'
    lock = TechnicianLock(path)
    lock.require()  # An uncommissioned API remains scriptable; HMI requires initial setup.
    with pytest.raises(ValueError):
        lock.setup('short')
    lock.setup('fixture-technical-pin')
    assert lock.unlocked
    assert 'fixture-technical-pin' not in path.read_text()
    lock = TechnicianLock(path)
    assert lock.configured and not lock.unlocked
    assert not lock.unlock('incorrect')
    with pytest.raises(PermissionError):
        lock.require()
    assert lock.unlock('fixture-technical-pin')
    lock.require()
    lock.lock()
    with pytest.raises(PermissionError):
        lock.require()
    with pytest.raises(RuntimeError):
        lock.setup('new-fixture-pin')
