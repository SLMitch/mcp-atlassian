"""Two-step file uploads for remote (HTTP) deployments.

A remote server cannot read the client's file paths, and inline base64 makes
the model transcribe the whole file. Instead the client asks for an upload slot
(``request``), PUTs the raw bytes to ``/uploads/{upload_id}`` with a single-use
token, then a tool consumes the staged file (``consume``).

Staged files live in memory-indexed temporary files on the local disk: run a
single replica (or session affinity / a shared ``MCP_UPLOAD_DIR``) so the PUT
and the follow-up tool call reach the same process.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_atlassian.utils.media import ATTACHMENT_MAX_BYTES

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_TTL_SECONDS = 600
UPLOAD_TOKEN_HEADER = "x-upload-token"  # noqa: S105 - header name, not a secret
_MAX_FILENAME_LENGTH = 255


class UploadError(Exception):
    """Upload failure carrying the HTTP status to answer with."""

    status_code = 400


class UploadNotFoundError(UploadError):
    status_code = 404


class UploadForbiddenError(UploadError):
    status_code = 403


class UploadConflictError(UploadError):
    status_code = 409


class UploadExpiredError(UploadError):
    status_code = 410


class UploadTooLargeError(UploadError):
    status_code = 413


@dataclass
class StagedUpload:
    """One upload slot: created by ``request``, filled by a single PUT."""

    upload_id: str
    token: str
    filename: str
    path: Path
    expires_at: float
    put_started: bool = False
    size: int | None = None
    content_type: str | None = None

    @property
    def received(self) -> bool:
        return self.size is not None


def validate_upload_filename(filename: str) -> str:
    """Return the filename if safe as an attachment name, else raise ValueError."""
    name = (filename or "").strip()
    if not name:
        raise ValueError("filename is required")
    if len(name) > _MAX_FILENAME_LENGTH:
        msg = f"filename longer than {_MAX_FILENAME_LENGTH} characters"
        raise ValueError(msg)
    if "/" in name or "\\" in name or "\x00" in name or name in {".", ".."}:
        raise ValueError("filename must be a plain name (no path, no '..')")
    return name


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        msg = f"{name} must be a positive integer, got {raw!r}"
        raise ValueError(msg)
    return value


class UploadStore:
    """In-memory registry of staged uploads backed by private temp files."""

    def __init__(
        self,
        directory: Path,
        max_bytes: int = ATTACHMENT_MAX_BYTES,
        ttl_seconds: int = DEFAULT_UPLOAD_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._uploads: dict[str, StagedUpload] = {}

    @classmethod
    def from_env(cls) -> UploadStore:
        """Build the store from MCP_UPLOAD_DIR / _MAX_BYTES / _TTL_SECONDS."""
        directory = os.getenv("MCP_UPLOAD_DIR", "").strip() or str(
            Path(tempfile.gettempdir()) / "mcp-atlassian-uploads"
        )
        return cls(
            directory=Path(directory),
            max_bytes=_env_int("MCP_UPLOAD_MAX_BYTES", ATTACHMENT_MAX_BYTES),
            ttl_seconds=_env_int("MCP_UPLOAD_TTL_SECONDS", DEFAULT_UPLOAD_TTL_SECONDS),
        )

    def request(self, filename: str) -> StagedUpload:
        """Open an upload slot for ``filename`` (validated, never used on disk)."""
        self.cleanup_expired()
        name = validate_upload_filename(filename)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        upload_id = secrets.token_urlsafe(16)
        upload = StagedUpload(
            upload_id=upload_id,
            token=secrets.token_urlsafe(32),
            filename=name,
            path=self.directory / f"{secrets.token_hex(16)}.upload",
            expires_at=self._clock() + self.ttl_seconds,
        )
        self._uploads[upload_id] = upload
        return upload

    def _get_live(self, upload_id: str) -> StagedUpload:
        upload = self._uploads.get(upload_id)
        if upload is None:
            msg = f"unknown upload_id {upload_id!r}"
            raise UploadNotFoundError(msg)
        if self._clock() >= upload.expires_at:
            self._discard(upload)
            msg = f"upload {upload_id!r} has expired"
            raise UploadExpiredError(msg)
        return upload

    async def receive(
        self,
        upload_id: str,
        token: str | None,
        chunks: AsyncIterator[bytes],
        content_length: int | None = None,
        content_type: str | None = None,
    ) -> int:
        """Stream the PUT body to disk; single use. Returns the stored size."""
        upload = self._get_live(upload_id)
        if not token or not hmac.compare_digest(
            token.encode("utf-8"), upload.token.encode("utf-8")
        ):
            raise UploadForbiddenError("invalid upload token")
        if upload.put_started:
            msg = f"upload {upload_id!r} was already sent"
            raise UploadConflictError(msg)
        # Any authenticated attempt consumes the slot, even a failed one.
        upload.put_started = True
        if content_length is not None and content_length > self.max_bytes:
            self._discard(upload)
            msg = f"file exceeds the {self.max_bytes} bytes limit"
            raise UploadTooLargeError(msg)

        size = 0
        fd = os.open(upload.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_bytes:
                        msg = f"file exceeds the {self.max_bytes} bytes limit"
                        raise UploadTooLargeError(msg)
                    out.write(chunk)
        except BaseException:
            self._discard(upload)
            raise
        upload.size = size
        upload.content_type = content_type
        return size

    def consume(self, upload_id: str) -> tuple[StagedUpload, bytes]:
        """Return the staged file's metadata and bytes, then delete it."""
        upload = self._get_live(upload_id)
        if not upload.received:
            msg = (
                f"upload {upload_id!r} has no file yet: PUT it to the upload URL first"
            )
            raise UploadConflictError(msg)
        try:
            data = upload.path.read_bytes()
        finally:
            self._discard(upload)
        return upload, data

    def cleanup_expired(self) -> int:
        """Drop expired slots and their files. Returns how many were removed."""
        now = self._clock()
        expired = [u for u in self._uploads.values() if now >= u.expires_at]
        for upload in expired:
            self._discard(upload)
        return len(expired)

    def _discard(self, upload: StagedUpload) -> None:
        self._uploads.pop(upload.upload_id, None)
        try:
            upload.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not delete staged upload {upload.path}: {e}")


_store: UploadStore | None = None


def get_upload_store() -> UploadStore:
    """Process-wide store shared by the PUT route and the tools."""
    global _store
    if _store is None:
        _store = UploadStore.from_env()
    return _store


def public_base_url(
    scheme: str,
    host: str | None,
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
) -> str:
    """Base URL clients use to reach this server (MCP_PUBLIC_BASE_URL wins).

    Raises:
        ValueError: If neither MCP_PUBLIC_BASE_URL nor a Host header is available.
    """
    configured = os.getenv("MCP_PUBLIC_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    host = (forwarded_host or "").split(",")[0].strip() or host
    if not host:
        raise ValueError(
            "Cannot build the upload URL: set MCP_PUBLIC_BASE_URL "
            "(e.g. https://mcp.example.com)"
        )
    proto = (forwarded_proto or scheme or "https").split(",")[0].strip()
    return f"{proto}://{host}".rstrip("/")


async def upload_put_endpoint(request: Request) -> JSONResponse:
    """``PUT /uploads/{upload_id}``: store the raw body of a staged upload.

    Outside the MCP endpoint, so the auth middleware does not cover it: the
    single-use token (``X-Upload-Token`` header or ``?token=``) is the guard.
    """
    upload_id = request.path_params["upload_id"]
    token = request.headers.get(UPLOAD_TOKEN_HEADER) or request.query_params.get(
        "token"
    )
    raw_length = request.headers.get("content-length")
    content_length = int(raw_length) if raw_length and raw_length.isdigit() else None
    try:
        size = await get_upload_store().receive(
            upload_id,
            token,
            request.stream(),
            content_length=content_length,
            content_type=request.headers.get("content-type"),
        )
    except UploadError as e:
        logger.info(f"Rejected upload {upload_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=e.status_code)
    return JSONResponse({"upload_id": upload_id, "size": size, "status": "received"})
