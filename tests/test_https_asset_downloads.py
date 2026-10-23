"""Actual TLS handshakes verify download trust, hostname, dates and payloads.

Certificates and private keys are generated only inside pytest's temporary
directory. No system trust stores are changed and no real assets are fetched.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ssl
import threading

import pytest

pytest.importorskip("cryptography")
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from src.models import weights


PAYLOAD = b"local HTTPS certificate-validation software fixture"


@pytest.fixture
def certificates(tmp_path):
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NeuroIris local test CA")])
    ca = (x509.CertificateBuilder()
          .subject_name(ca_name)
          .issuer_name(ca_name)
          .public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number())
          .not_valid_before(now - timedelta(days=2))
          .not_valid_after(now + timedelta(days=365))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, key_cert_sign=True, crl_sign=True,
                                       encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    ca_pem = ca.public_bytes(serialization.Encoding.PEM).decode("ascii")

    def issue(*, hostname="localhost", expired=False):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
        certificate = (x509.CertificateBuilder()
                       .subject_name(name)
                       .issuer_name(ca.subject)
                       .public_key(key.public_key())
                       .serial_number(x509.random_serial_number())
                       .not_valid_before(now - timedelta(days=1))
                       .not_valid_after(now - timedelta(hours=1) if expired else now + timedelta(days=1))
                       .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                       .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
                       .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                       .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                                                    key_encipherment=True, data_encipherment=False,
                                                    key_agreement=False, key_cert_sign=False, crl_sign=False,
                                                    encipher_only=False, decipher_only=False), critical=True)
                       .sign(ca_key, hashes.SHA256()))
        cert_path, key_path = tmp_path / "server.pem", tmp_path / "server.key"
        cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                               serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        return cert_path, key_path

    return ca_pem, issue


@contextmanager
def https_server(cert_path, key_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_port}/imagenette2.tgz", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "The local HTTPS test server did not stop"


def prepare_client(monkeypatch, ca_pem=None):
    context = weights.https_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    if ca_pem is not None:
        context.load_verify_locations(cadata=ca_pem)
    monkeypatch.setattr(weights, "https_context", lambda: context)
    # A developer's proxy must not intercept these strictly local handshakes.
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    return context


def test_trusted_local_ca_downloads_and_verifies_payload(tmp_path, monkeypatch, certificates):
    ca_pem, issue = certificates
    context = prepare_client(monkeypatch, ca_pem)
    target = tmp_path / "imagenette2.tgz"
    with https_server(*issue()) as (url, requests):
        result = weights.verified_download(url, target, hashlib.sha256(PAYLOAD).hexdigest())

    assert result == target
    assert target.read_bytes() == PAYLOAD
    assert requests == ["/imagenette2.tgz"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


@pytest.mark.parametrize("failure", ["untrusted_issuer", "wrong_hostname", "expired_certificate"])
def test_untrusted_hostname_and_expired_certificates_are_rejected(tmp_path, monkeypatch, certificates, failure):
    ca_pem, issue = certificates
    context = prepare_client(monkeypatch, None if failure == "untrusted_issuer" else ca_pem)
    certificate = issue(hostname="another-host.example" if failure == "wrong_hostname" else "localhost",
                        expired=failure == "expired_certificate")
    target = tmp_path / "imagenette2.tgz"
    messages = []
    with https_server(*certificate) as (url, requests):
        with pytest.raises(weights.ModelSetupError) as error:
            weights.verified_download(url, target, hashlib.sha256(PAYLOAD).hexdigest(), progress=messages.append)

    assert "certificat" in str(error.value).lower()
    assert not target.exists()
    assert not requests, "HTTP asset data must never be requested after a rejected TLS certificate"
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert not any("Retrying" in message for message in messages)


def test_valid_https_still_rejects_payload_with_wrong_sha256(tmp_path, monkeypatch, certificates):
    ca_pem, issue = certificates
    prepare_client(monkeypatch, ca_pem)
    target = tmp_path / "imagenette2.tgz"
    previous = b"existing cache must not be replaced by unverified data"
    target.write_bytes(previous)
    with https_server(*issue()) as (url, requests):
        with pytest.raises(weights.ModelSetupError, match="SHA256 mismatch"):
            weights.verified_download(url, target, "0" * 64)

    assert requests == ["/imagenette2.tgz"]
    assert target.read_bytes() == previous
