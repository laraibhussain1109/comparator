"""Interrupted HTTPS transfers must recover without publishing corrupt assets."""
from __future__ import annotations

import builtins
import errno
import hashlib
import io
import ssl
import time
import urllib.error

import pytest

from src.models import weights


ASSET_URL = "https://example.org/imagenette2.tgz"
PAYLOAD = b"official archive download fixture"


class Response(io.BytesIO):
    """Small HTTP response that can interrupt an otherwise valid body."""

    def __init__(self, payload, *, status=200, headers=None, fail_after=None, error=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), **(headers or {})}
        self.fail_after = fail_after
        self.error = error

    def read(self, size=-1):
        if self.fail_after is not None:
            remaining = self.fail_after - self.tell()
            if remaining <= 0:
                raise self.error
            size = remaining if size < 0 else min(size, remaining)
        return super().read(size)

    def getcode(self):
        return self.status

    def geturl(self):
        return ASSET_URL


@pytest.fixture(autouse=True)
def no_backoff_wait(monkeypatch):
    delays = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


def digest(payload=PAYLOAD):
    return hashlib.sha256(payload).hexdigest()


def install_responses(monkeypatch, responses):
    responses = iter(responses)
    requests = []

    def urlopen(request, **kwargs):
        requests.append((request, kwargs))
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(weights.urllib.request, "urlopen", urlopen)
    return requests


def request_headers(request):
    return {name.lower(): value for name, value in request.header_items()}


def assert_no_temporary_files(directory):
    assert not list(directory.glob(".dl-*.tmp"))


def test_ssl_record_failure_resumes_partial_transfer_with_validated_range(tmp_path, monkeypatch, no_backoff_wait):
    offset = 9
    etag = '"published-archive-version"'
    requests = install_responses(monkeypatch, [
        Response(PAYLOAD, headers={"ETag": etag}, fail_after=offset,
                 error=ssl.SSLError("DECRYPTION_FAILED_OR_BAD_RECORD_MAC")),
        Response(PAYLOAD[offset:], status=206, headers={
            "ETag": etag,
            "Content-Range": f"bytes {offset}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
        }),
    ])
    target = tmp_path / "imagenette2.tgz"
    messages = []

    assert weights.verified_download(ASSET_URL, target, digest(), progress=messages.append) == target

    assert target.read_bytes() == PAYLOAD
    assert len(requests) == 2
    assert request_headers(requests[1][0])["range"] == f"bytes={offset}-"
    assert request_headers(requests[1][0])["if-range"] == etag
    assert request_headers(requests[0][0])["accept-encoding"] == "identity"
    assert no_backoff_wait
    assert messages, "The training UI must receive asset-download status messages"
    # Native trust must retain certificate and hostname verification.
    contexts = [kwargs["context"] for _, kwargs in requests]
    assert contexts[0] is contexts[1]
    assert all(context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
               for context in contexts)
    assert_no_temporary_files(tmp_path)


def test_server_ignoring_range_restarts_without_appending_duplicate_bytes(tmp_path, monkeypatch):
    offset = 7
    requests = install_responses(monkeypatch, [
        Response(PAYLOAD, fail_after=offset, error=ConnectionResetError("connection reset")),
        Response(PAYLOAD, status=200),
    ])
    target = tmp_path / "imagenette2.tgz"

    weights.verified_download(ASSET_URL, target, digest())

    assert request_headers(requests[1][0])["range"] == f"bytes={offset}-"
    assert target.read_bytes() == PAYLOAD
    assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize("content_range", [
    "bytes 0-3/32",       # A resume response that starts at zero would duplicate bytes.
    "bytes 5-4/32",       # End before start.
    "bytes 5-32/32",      # End outside the complete resource.
    "bytes nonsense",    # Missing parseable bounds.
])
def test_invalid_partial_response_is_never_published(tmp_path, monkeypatch, content_range):
    requests = install_responses(monkeypatch, [
        Response(PAYLOAD, fail_after=5, error=ConnectionResetError("interrupted")),
        Response(PAYLOAD[5:], status=206, headers={"Content-Range": content_range}),
    ])
    target = tmp_path / "imagenette2.tgz"
    target.write_bytes(b"previous cache file")

    with pytest.raises(weights.ModelSetupError):
        weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) == 2
    assert target.read_bytes() == b"previous cache file"
    assert_no_temporary_files(tmp_path)


def test_premature_eof_retries_and_resumes_instead_of_reporting_checksum_failure(tmp_path, monkeypatch):
    offset = 8
    requests = install_responses(monkeypatch, [
        Response(PAYLOAD[:offset], headers={"Content-Length": str(len(PAYLOAD))}),
        Response(PAYLOAD[offset:], status=206, headers={
            "Content-Range": f"bytes {offset}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
        }),
    ])
    target = tmp_path / "imagenette2.tgz"

    weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) == 2
    assert target.read_bytes() == PAYLOAD
    assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize("status", [408, 429, 500, 503, 507])
def test_transient_http_errors_are_retried(tmp_path, monkeypatch, status):
    requests = install_responses(monkeypatch, [
        urllib.error.HTTPError(ASSET_URL, status, "temporary server error", {}, None),
        Response(PAYLOAD),
    ])
    target = tmp_path / "imagenette2.tgz"

    weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) == 2
    assert target.read_bytes() == PAYLOAD


@pytest.mark.parametrize("failure", [
    ssl.SSLCertVerificationError("certificate verify failed"),
    urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed")),
    urllib.error.HTTPError(ASSET_URL, 404, "Not Found", {}, None),
])
def test_certificate_rejection_and_missing_url_fail_without_retry(tmp_path, monkeypatch, failure, no_backoff_wait):
    requests = install_responses(monkeypatch, [failure])
    target = tmp_path / "imagenette2.tgz"

    with pytest.raises(weights.ModelSetupError):
        weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) == 1
    assert not no_backoff_wait
    assert not target.exists()
    assert_no_temporary_files(tmp_path)


def test_network_failure_retry_limit_preserves_destination(tmp_path, monkeypatch, no_backoff_wait):
    attempts = weights.DOWNLOAD_ATTEMPTS
    requests = install_responses(monkeypatch, [
        urllib.error.URLError(ssl.SSLError("DECRYPTION_FAILED_OR_BAD_RECORD_MAC"))
        for _ in range(attempts)
    ])
    target = tmp_path / "imagenette2.tgz"
    target.write_bytes(b"previous cache file")

    with pytest.raises(weights.ModelSetupError) as failure:
        weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) == attempts
    assert len(no_backoff_wait) == attempts - 1
    assert "imagenette2.tgz" in str(failure.value)
    assert target.read_bytes() == b"previous cache file"
    assert_no_temporary_files(tmp_path)


def test_disk_full_error_is_not_retried_as_network_failure(tmp_path, monkeypatch, no_backoff_wait):
    requests = install_responses(monkeypatch, [Response(PAYLOAD)])
    real_open = builtins.open

    def fail_write(path, mode="r", *args, **kwargs):
        if ".dl-" in str(path) and any(flag in mode for flag in "wax"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(weights, "open", fail_write, raising=False)
    target = tmp_path / "imagenette2.tgz"

    with pytest.raises((weights.ModelSetupError, OSError), match="No space left"):
        weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) <= 1
    assert not no_backoff_wait
    assert not target.exists()
    assert_no_temporary_files(tmp_path)


def test_corrupt_complete_body_never_replaces_existing_cache(tmp_path, monkeypatch):
    requests = install_responses(monkeypatch, [Response(b"incorrect archive")])
    target = tmp_path / "imagenette2.tgz"
    target.write_bytes(b"previous cache file")

    with pytest.raises(weights.ModelSetupError, match="SHA256 mismatch"):
        weights.verified_download(ASSET_URL, target, digest())

    assert len(requests) == 1
    assert target.read_bytes() == b"previous cache file"
    assert_no_temporary_files(tmp_path)


def test_verified_cache_skips_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("An already verified cache must work offline")

    monkeypatch.setattr(weights.urllib.request, "urlopen", no_network)
    target = tmp_path / "imagenette2.tgz"
    target.write_bytes(PAYLOAD)

    assert weights.verified_download(ASSET_URL, target, digest()) == target
    assert target.read_bytes() == PAYLOAD


def test_plain_http_asset_url_is_rejected_before_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Asset download must not use unverified HTTP")

    monkeypatch.setattr(weights.urllib.request, "urlopen", no_network)

    with pytest.raises(weights.ModelSetupError, match="HTTPS"):
        weights.verified_download("http://example.org/imagenette2.tgz", tmp_path / "asset.tgz", digest())


def test_redirect_to_plain_http_is_rejected_without_publishing(tmp_path, monkeypatch):
    response = Response(PAYLOAD)
    response.geturl = lambda: "http://example.org/imagenette2.tgz"
    install_responses(monkeypatch, [response])
    target = tmp_path / "imagenette2.tgz"

    with pytest.raises(weights.ModelSetupError, match="redirected"):
        weights.verified_download(ASSET_URL, target, digest())

    assert not target.exists()
    assert_no_temporary_files(tmp_path)
