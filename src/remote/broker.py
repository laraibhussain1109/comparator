"""Thread-safe handoff from the LAN server to the desktop's UI thread.

The server only sees published metadata. It never reads widgets or runs model
or inspection code. Commands become applied only after the PC reports success.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import re
import secrets
import threading
import time
from typing import Callable


DISPLAY_DECISIONS = {"PASS": "OK", "RECHECK": "View Recheck", "FAIL": "NG"}
REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class PhoneControlBroker:
    def __init__(self, *, command_ttl: float = 5.0, history_limit: int = 128,
                 clock: Callable[[], float] = time.monotonic):
        if command_ttl <= 0 or history_limit < 1:
            raise ValueError("Command lifetime and history size must be positive")
        self._clock = clock
        self._ttl = command_ttl
        self._history_limit = history_limit
        self._lock = threading.RLock()
        self._active = False
        self._published = False
        self._state: dict = {}
        self._fingerprint = None
        self._context_id = secrets.token_urlsafe(24)
        self._commands: OrderedDict[str, dict] = OrderedDict()
        self._pending: str | None = None

    def start(self):
        with self._lock:
            if not self._active:
                self._active = True
                self._published = False
                self._context_id = secrets.token_urlsafe(24)
                self._commands.clear()
                self._pending = None
        return self

    def stop(self):
        with self._lock:
            self._active = False
            self._published = False
            self._context_id = secrets.token_urlsafe(24)
            pending = self._commands.get(self._pending)
            if pending is not None and not pending['_claimed']:
                self._reject(pending, 503, "Phone control was stopped before this decision was recorded")

    def publish(self, state: dict):
        """Publish a fresh desktop snapshot; semantic changes revoke old context IDs."""
        with self._lock:
            current = deepcopy(state)
            private_context = current.pop('_context_key', None)
            current.pop('context_id', None)
            current.pop('last_command', None)
            fingerprint = (current, private_context)
            if self._fingerprint != fingerprint:
                self._context_id = secrets.token_urlsafe(24)
                self._fingerprint = deepcopy(fingerprint)
            self._state = current
            self._published = True
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            self._expire_queued()
            state = deepcopy(self._state)
            state['context_id'] = self._context_id
            if not self._active:
                state.update(can_submit=False, reason="Phone control is stopped")
            elif not self._published:
                state.update(can_submit=False, reason="Waiting for the PC's current inspection state")
            elif self._pending:
                state.update(can_submit=False, reason="The PC is recording a view decision")
            if self._commands:
                state['last_command'] = self._public(next(reversed(self._commands.values())))
            return 200, state

    def submit(self, payload: dict):
        with self._lock:
            self._expire_queued()
            if not isinstance(payload, dict):
                return self._error(400, "A verdict command must be a JSON object")
            request_id = payload.get('request_id')
            verdict = payload.get('verdict')
            context_id = payload.get('context_id')
            if not isinstance(request_id, str) or REQUEST_ID.fullmatch(request_id) is None:
                return self._error(400, "A valid request ID is required")
            if not isinstance(verdict, str) or verdict not in DISPLAY_DECISIONS:
                return self._error(400, "Verdict must be PASS, RECHECK or FAIL", request_id)
            if not isinstance(context_id, str) or not 1 <= len(context_id) <= 128:
                return self._error(400, "A current inspection context is required", request_id)
            existing = self._commands.get(request_id)
            if existing is not None:
                if existing['context_id'] != context_id or existing['verdict'] != verdict:
                    return self._error(409, "This request ID was already used for a different decision", request_id)
                return self._response(existing)
            if not self._active:
                return self._error(503, "Phone control is stopped", request_id)
            if context_id != self._context_id:
                return self._error(409, "The part, face or capture changed. Refresh the current inspection", request_id)
            if not self._published or self._state.get('can_submit') is not True:
                return self._error(409, str(self._state.get('reason') or "The PC is not ready for a manual decision"), request_id)
            if self._pending:
                return self._error(409, "Another view decision is already being recorded", request_id)
            command = dict(request_id=request_id, context_id=context_id, verdict=verdict,
                           status='pending', _created=self._clock(), _claimed=False, _code=202)
            self._commands[request_id] = command
            self._pending = request_id
            self._trim_history()
            return self._response(command)

    def take_next(self):
        """Called only on the UI thread after publishing its current state."""
        with self._lock:
            self._expire_queued()
            command = self._commands.get(self._pending)
            if command is None or command['_claimed']:
                return None
            if not self._active:
                self._reject(command, 503, "Phone control is stopped")
                return None
            if command['context_id'] != self._context_id:
                self._reject(command, 409, "The part, face or capture changed before this tap reached the PC")
                return None
            if not self._published or self._state.get('can_submit') is not True:
                self._reject(command, 409, str(self._state.get('reason') or "The PC is no longer ready"))
                return None
            command['_claimed'] = True
            return {key: command[key] for key in ('request_id', 'context_id', 'verdict')}

    def complete(self, request_id: str, success: bool, verdict=None, error=None):
        with self._lock:
            command = self._commands.get(request_id)
            if command is None:
                return self._error(404, "This command is no longer known to the PC", request_id)
            if command['status'] != 'pending':
                return self._response(command)
            if success and command['_claimed'] and (verdict is None or verdict == command['verdict']):
                command.update(status='applied', _code=200,
                               display_decision=DISPLAY_DECISIONS[command['verdict']])
                self._pending = None
            else:
                self._reject(command, 409, str(error or "The PC could not record this decision"))
            return self._response(command)

    def command_status(self, request_id: str):
        with self._lock:
            self._expire_queued()
            if not isinstance(request_id, str) or REQUEST_ID.fullmatch(request_id) is None:
                return self._error(400, "A valid request ID is required")
            command = self._commands.get(request_id)
            if command is None:
                return self._error(404, "This command is no longer known to the PC", request_id)
            return self._response(command)

    def _expire_queued(self):
        command = self._commands.get(self._pending)
        if command is not None and not command['_claimed'] and self._clock() - command['_created'] > self._ttl:
            self._reject(command, 410, "The tap expired before the PC could record it. Refresh and try again")

    def _reject(self, command, code, message):
        command.update(status='rejected', error=message, _code=code)
        if self._pending == command['request_id']:
            self._pending = None

    @staticmethod
    def _public(command):
        return {key: deepcopy(value) for key, value in command.items()
                if not key.startswith('_') and key != 'context_id'}

    def _response(self, command):
        return command['_code'], self._public(command)

    @staticmethod
    def _error(code, message, request_id=None):
        result = dict(status='rejected', error=message)
        if request_id is not None:
            result['request_id'] = request_id
        return code, result

    def _trim_history(self):
        while len(self._commands) > self._history_limit:
            oldest = next(iter(self._commands))
            if oldest == self._pending:
                break
            self._commands.pop(oldest)
