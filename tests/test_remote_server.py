"""Exercise the LAN listener and authentication over actual local HTTP sockets."""
from http.client import HTTPConnection, HTTPResponse
import json
import socket
import threading
import time

import pytest

import src.remote.server as transport
from src.remote.server import PhoneControlServer, local_ipv4_addresses


class Broker:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.submissions = []
        self.threads = []

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def snapshot(self):
        self.threads.append(threading.get_ident())
        return 200, {'enabled': True, 'context_id': 'context-1', 'part_id': 'housing-1'}

    def submit(self, payload):
        self.threads.append(threading.get_ident())
        self.submissions.append(payload)
        return 202, {'request_id': payload['request_id'], 'status': 'queued'}

    def command_status(self, request_id):
        self.threads.append(threading.get_ident())
        if request_id == 'missing':
            return 404, {'error': 'Unknown command'}
        return 200, {'request_id': request_id, 'status': 'applied'}


@pytest.fixture
def server(tmp_path, monkeypatch):
    for name, content in [('index.html', '<html>Phone control</html>'),
                          ('control.js', '"use strict";'), ('control.css', 'body { color: green; }')]:
        (tmp_path / name).write_text(content, encoding='utf-8')
    monkeypatch.setattr(transport, '_WEB_ROOT', tmp_path)
    instance = PhoneControlServer(Broker(), host='127.0.0.1', port=0).start()
    try:
        yield instance
    finally:
        instance.stop()


def request(server, path='/api/state', *, method='GET', body=None, headers=None, auth=True):
    connection = HTTPConnection('127.0.0.1', server.port, timeout=2)
    supplied = {'Authorization': 'Bearer ' + server.token} if auth else {}
    supplied.update(headers or {})
    if isinstance(body, dict):
        body = json.dumps(body)
        supplied.setdefault('Content-Type', 'application/json')
    try:
        connection.request(method, path, body=body, headers=supplied)
        response = connection.getresponse()
        content = response.read()
        if response.getheader('Content-Type', '').startswith('application/json'):
            content = json.loads(content)
        return response.status, dict(response.getheaders()), content
    finally:
        connection.close()


def test_explicit_start_stop_rotates_pairing_token_and_releases_listener():
    broker = Broker()
    server = PhoneControlServer(broker, host='127.0.0.1', port=0)
    assert not server.running and server.token is None and broker.starts == 0
    with pytest.raises(RuntimeError, match='Start Phone Control'):
        server.pairing_url('127.0.0.1')
    server.start()
    port, token = server.port, server.token
    assert token and len(token) >= 32 and server.running and broker.starts == 1
    assert server.pairing_url('127.0.0.1') == f'http://127.0.0.1:{port}/#token={token}'
    assert server.start() is server and broker.starts == 1
    server.stop()
    assert not server.running and server.token is None and broker.stops == 1
    server.stop()
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1', port), timeout=.2)
    try:
        server.start()
        assert server.token != token and broker.starts == 2
        status, _, _ = request(server, headers={'Authorization': 'Bearer ' + token})
        assert status == 401
    finally:
        server.stop()


def test_failed_bind_does_not_enable_broker():
    broker = Broker()
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        server = PhoneControlServer(broker, host='127.0.0.1', port=occupied.getsockname()[1])
        with pytest.raises(OSError):
            server.start()
    assert broker.starts == broker.stops == 0
    assert not server.running and server.token is None


def test_broker_stop_failure_still_closes_listener(server):
    port = server.port

    def fail_stop():
        raise RuntimeError('Controlled broker failure')

    server.broker.stop = fail_stop
    with pytest.raises(RuntimeError, match='Controlled broker failure'):
        server.stop()
    assert not server.running and server.token is None
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1', port), timeout=.2)


@pytest.mark.parametrize('path,kind', [('/', 'text/html'), ('/control.js', 'text/javascript'),
                                       ('/control.css', 'text/css')])
def test_only_fixed_static_assets_are_public_and_never_cached(server, path, kind):
    status, headers, content = request(server, path, auth=False)
    assert status == 200 and content
    assert headers['Content-Type'].startswith(kind)
    assert headers['Cache-Control'] == 'no-store'
    assert "script-src 'self'" in headers['Content-Security-Policy']
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    assert headers['Referrer-Policy'] == 'no-referrer'
    assert 'Access-Control-Allow-Origin' not in headers
    assert request(server, '/../server.py')[0] == 404
    assert request(server, '/control.js?anything=1')[0] == 404


def test_bearer_auth_required_for_state_submission_and_command_status(server):
    payload = {'request_id': 'command-1', 'context_id': 'context-1', 'verdict': 'PASS'}
    assert request(server, auth=False)[0] == 401
    assert request(server, headers={'Authorization': 'Bearer old-pairing'})[0] == 401
    assert request(server, '/api/commands/command-1', auth=False)[0] == 401
    assert request(server, '/api/verdict', method='POST', body=payload, auth=False)[0] == 401
    assert server.broker.submissions == []


def test_authenticated_commands_use_broker_tuples_from_worker_threads(server):
    status, _, state = request(server)
    assert status == 200 and state['context_id'] == 'context-1'
    payload = {'request_id': 'command-1', 'context_id': 'context-1', 'verdict': 'RECHECK'}
    status, _, queued = request(server, '/api/verdict', method='POST', body=payload,
                                 headers={'Origin': f'http://127.0.0.1:{server.port}'})
    assert status == 202 and queued['status'] == 'queued'
    assert server.broker.submissions == [payload]
    status, _, applied = request(server, '/api/commands/command-1')
    assert status == 200 and applied['status'] == 'applied'
    assert request(server, '/api/commands/missing')[0] == 404
    assert all(identity != threading.get_ident() for identity in server.broker.threads)


@pytest.mark.parametrize('origin', ['http://attacker.example', 'null', 'https://127.0.0.1',
                                     'http://127.0.0.1:1', 'http://localhost:8765'])
def test_cross_origin_commands_are_rejected_before_broker(server, origin):
    status, _, _ = request(server, '/api/verdict', method='POST',
                            body={'request_id': 'command-1'}, headers={'Origin': origin})
    assert status == 403 and not server.broker.submissions


@pytest.mark.parametrize('host', ['attacker.example', 'localhost.evil:8765', '8.8.8.8:8765',
                                   '127.0.0.1:1', 'user@127.0.0.1:8765', '127.0.0.1/path'])
def test_untrusted_host_headers_are_rejected(server, host):
    assert request(server, headers={'Host': host})[0] == 403


@pytest.mark.parametrize('body,headers,expected', [
    ('{}', {'Content-Type': 'text/plain'}, 415),
    ('[]', {'Content-Type': 'application/json'}, 400),
    ('{', {'Content-Type': 'application/json'}, 400),
    ('{"verdict":NaN}', {'Content-Type': 'application/json'}, 400),
    ('{}', {'Content-Type': 'application/json', 'Content-Length': '4097'}, 413),
    ('{}', {'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked'}, 400),
    ('{}', {'Content-Type': 'application/json', 'Content-Length': '-1'}, 400),
])
def test_invalid_or_oversized_command_bodies_never_reach_broker(server, body, headers, expected):
    status, _, _ = request(server, '/api/verdict', method='POST', body=body, headers=headers)
    assert status == expected and not server.broker.submissions


def test_duplicate_host_and_body_length_headers_are_rejected(server):
    for field, value in [('Host', f'127.0.0.1:{server.port}'), ('Content-Length', '2')]:
        connection = HTTPConnection('127.0.0.1', server.port, timeout=2)
        try:
            connection.putrequest('POST', '/api/verdict')
            connection.putheader('Authorization', 'Bearer ' + server.token)
            connection.putheader('Content-Type', 'application/json')
            connection.putheader('Content-Length', '2')
            connection.putheader(field, value)
            connection.endheaders(b'{}')
            response = connection.getresponse()
            assert response.status in {400, 403}
            response.read()
        finally:
            connection.close()
    assert not server.broker.submissions


def test_errors_and_request_logs_never_expose_token(server, capsys):
    token = server.token
    assert request(server, '/not-found?token=' + token)[0] == 404

    def fail():
        raise RuntimeError('secret=' + token)

    server.broker.snapshot = fail
    status, _, payload = request(server)
    assert status == 500 and token not in json.dumps(payload)
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err


def test_stop_closes_stalled_request_connections_without_waiting_for_timeout(server):
    connection = socket.create_connection(('127.0.0.1', server.port), timeout=2)
    connection.sendall(b'GET / HTTP/1.1\r\nHost: ')
    deadline = time.monotonic() + 1
    while not server._server._connections and time.monotonic() < deadline:
        time.sleep(.005)
    started = time.monotonic()
    server.stop()
    assert time.monotonic() - started < 1
    try:
        assert connection.recv(1024) == b''
    except ConnectionResetError:
        pass
    finally:
        connection.close()


def test_partial_body_has_bounded_read_timeout():
    server = PhoneControlServer(Broker(), host='127.0.0.1', port=0, request_timeout=.1).start()
    connection = socket.create_connection(('127.0.0.1', server.port), timeout=2)
    try:
        headers = (f'POST /api/verdict HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n'
                   f'Authorization: Bearer {server.token}\r\nContent-Type: application/json\r\n'
                   'Content-Length: 100\r\n\r\n{')
        connection.sendall(headers.encode('ascii'))
        response = HTTPResponse(connection)
        response.begin()
        assert response.status == 408
        response.read()
    finally:
        connection.close()
        server.stop()


def test_only_private_and_loopback_peers_can_reach_handlers(server):
    assert server._server.verify_request(None, ('127.0.0.1', 1))
    assert server._server.verify_request(None, ('192.168.10.2', 1))
    assert not server._server.verify_request(None, ('8.8.8.8', 1))
    assert not server._server.verify_request(None, ('0.0.0.0', 1))


def test_address_discovery_uses_hostname_and_filters_public_and_duplicate_addresses(monkeypatch):
    seen = []

    def discover(hostname, *args):
        seen.append(hostname)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 0)) for address in
                ('192.168.1.8', '10.0.0.4', '127.0.0.1', '8.8.8.8', '192.168.1.8')]

    monkeypatch.setattr(socket, 'gethostname', lambda: 'inspection-station')
    monkeypatch.setattr(socket, 'getaddrinfo', discover)
    assert local_ipv4_addresses() == ['10.0.0.4', '192.168.1.8']
    assert seen == ['inspection-station']
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args: [])
    assert local_ipv4_addresses() == ['127.0.0.1']
