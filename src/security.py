"""Local technician settings lock, intended for a single workstation HMI.

This prevents accidental operator changes. It is not an operating-system access
boundary: someone with write access to the runtime directory can reset it.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path

from src.artifacts import atomic_json
from src.utils.filesystem import filesystem_path


class TechnicianLock:
    ITERATIONS = 600_000

    def __init__(self, path: Path):
        self.path = filesystem_path(path)
        self.unlocked = False

    @property
    def configured(self):
        return self.path.is_file()

    def setup(self, pin: str):
        if self.configured:
            raise RuntimeError('A technician PIN is already configured; unlock it to change protected settings')
        if not isinstance(pin, str) or len(pin) < 6:
            raise ValueError('Use at least 6 characters for the technician PIN')
        salt = os.urandom(32)
        hashed = hashlib.pbkdf2_hmac('sha256', pin.encode('utf-8'), salt, self.ITERATIONS)
        atomic_json(self.path, {'schema_version': 1, 'salt': salt.hex(), 'iterations': self.ITERATIONS, 'hash': hashed.hex()})
        try:
            self.path.chmod(0o600)
        except OSError:
            pass  # Windows permissions are managed by the workstation account.
        self.unlocked = True

    def unlock(self, pin: str):
        if not self.configured:
            return False
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or data.get('iterations') != self.ITERATIONS:
            raise ValueError('Unsupported technician lock format')
        computed = hashlib.pbkdf2_hmac('sha256', pin.encode('utf-8'), bytes.fromhex(data['salt']), data['iterations'])
        self.unlocked = hmac.compare_digest(computed.hex(), data['hash'])
        return self.unlocked

    def lock(self):
        self.unlocked = False

    def require(self):
        if self.configured and not self.unlocked:
            raise PermissionError('Technician mode is locked. Unlock the HMI before changing advanced settings.')
