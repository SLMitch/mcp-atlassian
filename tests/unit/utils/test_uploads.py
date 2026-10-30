"""Tests for two-step staged uploads (utils/uploads.py)."""

import stat
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from mcp_atlassian.utils import uploads
from mcp_atlassian.utils.uploads import (
    UploadConflictError,
    UploadExpiredError,
    UploadForbiddenError,
    UploadNotFoundError,
    UploadStore,
    UploadTooLargeError,
    public_base_url,
    upload_put_endpoint,
    validate_upload_filename,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path: Path, clock: FakeClock) -> UploadStore:
    return UploadStore(tmp_path / "up", max_bytes=10, ttl_seconds=60, clock=clock)


@pytest.mark.anyio
async def test_request_receive_consume(store: UploadStore) -> None:
    upload = store.request("diagram.png")
    size = await store.receive(upload.upload_id, upload.token, _chunks(b"abc", b"de"))
    assert size == 5
    # Private file under a random name, never the client-supplied one.
    assert upload.path.name != "diagram.png"
    assert stat.S_IMODE(upload.path.stat().st_mode) == 0o600

    meta, data = store.consume(upload.upload_id)
    assert data == b"abcde"
    assert meta.filename == "diagram.png"
    assert not upload.path.exists()
    with pytest.raises(UploadNotFoundError):
        store.consume(upload.upload_id)


@pytest.mark.anyio
async def test_wrong_token_is_rejected(store: UploadStore) -> None:
    upload = store.request("a.txt")
    with pytest.raises(UploadForbiddenError):
        await store.receive(upload.upload_id, "wrong", _chunks(b"x"))
    with pytest.raises(UploadForbiddenError):
        await store.receive(upload.upload_id, None, _chunks(b"x"))
    # A rejected token does not burn the slot.
    assert await store.receive(upload.upload_id, upload.token, _chunks(b"x")) == 1


@pytest.mark.anyio
async def test_too_large_by_header_and_by_stream(store: UploadStore) -> None:
    by_header = store.request("a.bin")
    with pytest.raises(UploadTooLargeError):
        await store.receive(
            by_header.upload_id, by_header.token, _chunks(b"x"), content_length=11
        )

    by_stream = store.request("b.bin")
    with pytest.raises(UploadTooLargeError):
        await store.receive(
            by_stream.upload_id, by_stream.token, _chunks(b"123456", b"789012")
        )
    assert not by_stream.path.exists()
    with pytest.raises(UploadNotFoundError):
        store.consume(by_stream.upload_id)


@pytest.mark.anyio
async def test_second_put_conflicts(store: UploadStore) -> None:
    upload = store.request("a.txt")
    await store.receive(upload.upload_id, upload.token, _chunks(b"1"))
    with pytest.raises(UploadConflictError):
        await store.receive(upload.upload_id, upload.token, _chunks(b"2"))


@pytest.mark.anyio
async def test_expired_upload(store: UploadStore, clock: FakeClock) -> None:
    upload = store.request("a.txt")
    await store.receive(upload.upload_id, upload.token, _chunks(b"1"))
    clock.now += 61
    with pytest.raises(UploadExpiredError):
        store.consume(upload.upload_id)
    assert not upload.path.exists()


@pytest.mark.anyio
async def test_cleanup_expired(store: UploadStore, clock: FakeClock) -> None:
    upload = store.request("a.txt")
    await store.receive(upload.upload_id, upload.token, _chunks(b"1"))
    clock.now += 61
    assert store.cleanup_expired() == 1
    assert not upload.path.exists()


def test_consume_before_put_conflicts(store: UploadStore) -> None:
    upload = store.request("a.txt")
    with pytest.raises(UploadConflictError, match="PUT it"):
        store.consume(upload.upload_id)


@pytest.mark.parametrize("bad", ["", "  ", "../x", "a/b", "a\\b", "..", "x\x00"])
def test_invalid_filenames(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_upload_filename(bad)


def test_public_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_PUBLIC_BASE_URL", raising=False)
    assert public_base_url("http", "mcp.example.com", "https") == (
        "https://mcp.example.com"
    )
    assert (
        public_base_url("http", "backend:8000", "https", "mcp.example.com")
        == "https://mcp.example.com"
    )
    with pytest.raises(ValueError, match="MCP_PUBLIC_BASE_URL"):
        public_base_url("", None)
    monkeypatch.setenv("MCP_PUBLIC_BASE_URL", "https://pub.example.com/")
    assert public_base_url("http", "ignored") == "https://pub.example.com"


def test_put_endpoint(store: UploadStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uploads, "_store", store)
    app = Starlette(
        routes=[Route("/uploads/{upload_id}", upload_put_endpoint, methods=["PUT"])]
    )
    client = TestClient(app)
    upload = store.request("a.txt")
    url = f"/uploads/{upload.upload_id}"

    assert (
        client.put(url, content=b"hi", headers={"X-Upload-Token": "no"}).status_code
        == 403
    )
    ok = client.put(url, content=b"hi", headers={"X-Upload-Token": upload.token})
    assert ok.status_code == 200
    assert ok.json() == {"upload_id": upload.upload_id, "size": 2, "status": "received"}
    again = client.put(f"{url}?token={upload.token}", content=b"hi")
    assert again.status_code == 409
    assert client.put("/uploads/unknown", content=b"x").status_code == 404

    big = store.request("big.bin")
    too_big = client.put(
        f"/uploads/{big.upload_id}?token={big.token}", content=b"x" * 11
    )
    assert too_big.status_code == 413
