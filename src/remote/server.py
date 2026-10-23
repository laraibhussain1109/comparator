"""Small, explicitly started LAN transport; inspection stays on the GUI thread.

The broker is the only application boundary. Its ``snapshot()``,
``submit(payload)`` and ``command_status(request_id)`` methods return an HTTP
status and JSON object, and must be thread safe without accessing Qt objects.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import secrets
import socket
import threading
from urllib.parse import urlsplit


_WEB_ROOT = Path(__file__).with_name('web')
_ASSETS = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/control.js': ('control.js', 'text/javascript; charset=utf-8'),
    '/control.css': ('control.css', 'text/css; charset=utf-8'),
}
_REQUEST_ID = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
_MAX_BODY = 4096
_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'; object-src 'none'")


def _local_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return (address.is_private or address.is_loopback) and not (
        address.is_unspecified or address.is_multicast or address.is_reserved)


def local_ipv4_addresses() -> list[str]:
    """Discover local interfaces through the hostname, without internet probes."""
    addresses = set()
    try:
        entries = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET,
                                     socket.SOCK_STREAM)
    except OSError:
        entries = []
    for entry in entries:
        value = entry[4][0]
        if _local_address(value) and not ipaddress.ip_address(value).is_loopback:
            addresses.add(value)
    return sorted(addresses, key=ipaddress.ip_address) or ['127.0.0.1']


def _authority(value: str) -> tuple[str, int] | None:
    """Accept numeric LAN hosts and localhost without resolving untrusted DNS."""
    if not value or any(character.isspace() for character in value):
        return None
    try:
        parsed = urlsplit('//' + value)
        if (parsed.path or parsed.query or parsed.fragment or parsed.username is not None
                or parsed.password is not None or not parsed.hostname):
            return None
        host, port = parsed.hostname, parsed.port if parsed.port is not None else 80
        if host.lower() != 'localhost':
            if '%' in host or not _local_address(host):
                return None
            host = str(ipaddress.ip_address(host))
        if not 1 <= port <= 65535:
            return None
        return host.lower(), port
    except ValueError:
        return None


class _LANHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(self, address, broker, token, request_timeout):
        self.broker = broker
        self.token = token
        self.request_timeout = request_timeout
        self._connections = set()
        self._connections_lock = threading.Lock()
        super().__init__(address, _RequestHandler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.request_timeout)
        with self._connections_lock:
            self._connections.add(connection)
        return connection, address

    def verify_request(self, request, client_address):
        return _local_address(client_address[0])

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._connections_lock:
                self._connections.discard(request)

    def close_connections(self):
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def handle_error(self, request, client_address):
        # Never print request paths, authorization headers, or payloads.
        pass


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = 'NeuroIrisPhoneControl'
    sys_version = ''

    def log_message(self, format, *args):
        # BaseHTTPRequestHandler logs raw paths (possibly including secrets).
        pass

    def _respond(self, status, payload, content_type='application/json; charset=utf-8'):
        body = (json.dumps(payload, ensure_ascii=True, allow_nan=False).encode('utf-8')
                if isinstance(payload, dict) else payload)
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Security-Policy', _CSP)
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Connection', 'close')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)
        except (OSError, ConnectionError):
            pass

    def send_error(self, code, message=None, explain=None):
        self._respond(code, {'error': 'Request could not be processed'})

    def _check_host(self):
        values = self.headers.get_all('Host', [])
        host = _authority(values[0]) if len(values) == 1 else None
        if host is None or host[1] != self.server.server_port:
            self._respond(403, {'error': 'Use the local phone control address'})
            return False
        return True

    def _authenticate(self):
        headers = self.headers.get_all('Authorization', [])
        supplied = headers[0].encode('utf-8') if len(headers) == 1 else b''
        expected = ('Bearer ' + self.server.token).encode('ascii')
        if not self.server.token or not secrets.compare_digest(supplied, expected):
            self._respond(401, {'error': 'Pair this phone using the current desktop link'})
            return False
        return True

    def _same_origin(self):
        origins = self.headers.get_all('Origin', [])
        if not origins:
            return True
        if len(origins) == 1:
            try:
                origin = urlsplit(origins[0])
                if (origin.scheme == 'http' and not origin.path and not origin.query
                        and not origin.fragment and _authority(origin.netloc) ==
                        _authority(self.headers['Host'])):
                    return True
            except ValueError:
                pass
        self._respond(403, {'error': 'Phone commands must come from this control page'})
        return False

    def _broker_response(self, operation, *args):
        try:
            status, payload = operation(*args)
            if not isinstance(status, int) or not 200 <= status <= 599 or not isinstance(payload, dict):
                raise ValueError('Invalid broker response')
            self._respond(status, payload)
        except Exception:
            self._respond(500, {'error': 'Phone control is temporarily unavailable'})

    def do_GET(self):
        if not self._check_host():
            return
        if self.path in _ASSETS:
            name, content_type = _ASSETS[self.path]
            try:
                content = (_WEB_ROOT / name).read_bytes()
            except OSError:
                self._respond(503, {'error': 'Phone control page is unavailable'})
                return
            self._respond(200, content, content_type)
            return
        if not self._authenticate():
            return
        if self.path == '/api/state':
            self._broker_response(self.server.broker.snapshot)
            return
        prefix = '/api/commands/'
        if self.path.startswith(prefix) and _REQUEST_ID.fullmatch(self.path[len(prefix):]):
            self._broker_response(self.server.broker.command_status, self.path[len(prefix):])
            return
        self._respond(404, {'error': 'Unknown phone control resource'})

    def do_POST(self):
        if not self._check_host() or not self._same_origin() or not self._authenticate():
            return
        if self.path != '/api/verdict':
            self._respond(404, {'error': 'Unknown phone control resource'})
            return
        lengths = self.headers.get_all('Content-Length', [])
        if (self.headers.get_all('Transfer-Encoding', []) or len(lengths) != 1
                or len(lengths[0]) > 10 or not lengths[0].isascii() or not lengths[0].isdigit()):
            self._respond(400, {'error': 'A single Content-Length is required'})
            return
        length = int(lengths[0])
        if length > _MAX_BODY:
            self._respond(413, {'error': 'Phone command is too large'})
            return
        types = self.headers.get_all('Content-Type', [])
        if len(types) != 1 or types[0].split(';', 1)[0].strip().lower() != 'application/json':
            self._respond(415, {'error': 'Phone commands must use application/json'})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete command')
            payload = json.loads(body.decode('utf-8'), parse_constant=self._invalid_json_constant)
            if not isinstance(payload, dict):
                raise ValueError('A command object is required')
        except (TimeoutError, socket.timeout):
            self._respond(408, {'error': 'Phone command timed out'})
            return
        except (ValueError, UnicodeDecodeError):
            self._respond(400, {'error': 'Phone command must be a complete JSON object'})
            return
        self._broker_response(self.server.broker.submit, payload)

    @staticmethod
    def _invalid_json_constant(value):
        raise ValueError('Non-finite JSON value')


class PhoneControlServer:
    """Own the listener, pairing token and explicit request-thread shutdown."""

    def __init__(self, broker, host='0.0.0.0', port=8765, request_timeout=5.0):
        if not 0 <= int(port) <= 65535 or request_timeout <= 0:
            raise ValueError('Use a valid port and positive request timeout')
        self.broker = broker
        self.host = host
        self._configured_port = int(port)
        self.request_timeout = float(request_timeout)
        self._server = None
        self._thread = None
        self._lock = threading.RLock()

    @property
    def running(self):
        with self._lock:
            return self._server is not None

    @property
    def port(self):
        with self._lock:
            return self._server.server_port if self._server else self._configured_port

    @property
    def token(self):
        with self._lock:
            return self._server.token if self._server else None

    def start(self):
        with self._lock:
            if self._server is not None:
                return self
            server = _LANHTTPServer((self.host, self._configured_port), self.broker,
                                    secrets.token_urlsafe(32), self.request_timeout)
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .1},
                                      name='NeuroIrisPhoneControl', daemon=True)
            try:
                self.broker.start()
                thread.start()
            except Exception:
                try:
                    self.broker.stop()
                finally:
                    server.server_close()
                raise
            self._server, self._thread = server, thread
            return self

    def stop(self):
        with self._lock:
            if self._server is None:
                return
            server, thread = self._server, self._thread
            try:
                self.broker.stop()
            finally:
                server.token = ''
                try:
                    server.shutdown()
                    server.close_connections()
                    server.server_close()
                    thread.join()
                finally:
                    self._server = self._thread = None

    def pairing_url(self, address):
        with self._lock:
            if self._server is None:
                raise RuntimeError('Start Phone Control before pairing a phone')
            if address != 'localhost' and not _local_address(address):
                raise ValueError('Use a local interface IP address')
            host = '[' + address + ']' if ':' in address else address
            return f'http://{host}:{self._server.server_port}/#token={self._server.token}'
