"""Shared media type detection and attachment download utilities."""

import base64
import binascii
import json
import logging
import mimetypes
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Maximum attachment size for inline download (50 MB).
# Used by both Jira and Confluence server tools to gate in-memory transfers.
ATTACHMENT_MAX_BYTES: int = 50 * 1024 * 1024

_IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "image/bmp",
    }
)

# MIME types that carry no reliable information about the actual file type,
# so detection falls back to the filename extension. "multipart/form-data" is
# what Jira Service Management reports for attachments uploaded through the
# customer portal, which is the usual source of screenshots on support tickets.
_AMBIGUOUS_MIME_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/binary",
        "multipart/form-data",
    }
)

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}
)


def is_image_attachment(
    media_type: str | None, filename: str | None
) -> tuple[bool, str]:
    """Detect whether an attachment is an image.

    Uses two-tier detection: explicit MIME type check, then filename
    extension fallback for ambiguous or missing MIME types.

    Args:
        media_type: The MIME type reported by the API.
        filename: The attachment filename.

    Returns:
        Tuple of (is_image, resolved_mime_type).
    """
    if media_type and media_type in _IMAGE_MIME_TYPES:
        return True, media_type

    if (media_type in _AMBIGUOUS_MIME_TYPES or media_type is None) and filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in _IMAGE_EXTENSIONS:
            guessed = mimetypes.guess_type(filename)[0] or "image/png"
            return True, guessed

    return False, media_type or "application/octet-stream"


def fetch_and_encode_attachment(
    fetch_fn: Callable[[str], bytes | None],
    url: str,
    filename: str,
    mime_type: str | None = None,
    max_bytes: int = ATTACHMENT_MAX_BYTES,
) -> tuple[str | None, str | None, int]:
    """Fetch and base64-encode an attachment.

    Handles size-limit checks, fetching, encoding, and MIME type
    resolution in one place.

    Args:
        fetch_fn: Callable that takes a URL and returns raw bytes,
            or None on failure.
        url: The URL to fetch the attachment from.
        filename: The filename for MIME type detection fallback.
        mime_type: Explicit MIME type. When None the type is guessed
            from *filename* with ``application/octet-stream`` as the
            fallback.
        max_bytes: Maximum allowed file size in bytes.

    Returns:
        A 3-tuple ``(base64_data, resolved_mime_type, fetched_bytes)``.

        On success all three fields are populated.  On failure the
        first two are ``None`` and *fetched_bytes* distinguishes
        the failure mode:

        * ``fetched_bytes == 0`` -- fetch returned ``None`` or
          raised an exception.
        * ``fetched_bytes > 0``  -- downloaded data exceeded
          *max_bytes* (the actual size is returned so callers can
          report it).
    """
    try:
        data_bytes = fetch_fn(url)
    except Exception:
        logger.warning(
            "Failed to fetch attachment '%s' from %s",
            filename,
            url,
            exc_info=True,
        )
        return None, None, 0

    if data_bytes is None:
        logger.warning(
            "Fetch returned None for attachment '%s'",
            filename,
        )
        return None, None, 0

    actual_size = len(data_bytes)

    if actual_size > max_bytes:
        logger.warning(
            "Attachment '%s' fetched size %d exceeds limit %d",
            filename,
            actual_size,
            max_bytes,
        )
        return None, None, actual_size

    encoded = base64.b64encode(data_bytes).decode("ascii")

    if mime_type is None:
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    return encoded, mime_type, actual_size


def parse_base64_attachments(
    attachments_base64: str | None,
) -> list[dict[str, object]]:
    """Parse and decode attachments supplied as base64 content.

    Each entry must be an object with a ``filename`` and base64-encoded
    ``content_base64``. Decoding happens here so the client layer only ever
    deals with raw bytes, never with the server's filesystem.

    Args:
        attachments_base64: JSON array string of attachment objects, or None

    Returns:
        A list of ``{"filename": str, "content": bytes}`` dicts, empty if
        nothing was supplied

    Raises:
        ValueError: If the JSON, an entry or a filename is invalid, or if the
            content is undecodable, empty, or over ``ATTACHMENT_MAX_BYTES``
    """
    stripped = (attachments_base64 or "").strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        msg = f"attachments_base64 is not valid JSON: {e}"
        raise ValueError(msg) from e

    if not isinstance(parsed, list):
        raise ValueError(
            "attachments_base64 must be a JSON array of attachment objects."
        )

    decoded: list[dict[str, object]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(
                f"attachments_base64[{index}] must be an object with "
                "'filename' and 'content_base64'."
            )

        filename = item.get("filename")
        if not filename or not isinstance(filename, str):
            raise ValueError(f"attachments_base64[{index}] requires a 'filename'.")

        content_base64 = item.get("content_base64")
        if not isinstance(content_base64, str):
            raise ValueError(
                f"attachments_base64[{index}] ({filename}) requires 'content_base64'."
            )

        try:
            content = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as e:
            msg = (
                f"attachments_base64[{index}] ({filename}) has invalid "
                f"base64 content: {e}"
            )
            raise ValueError(msg) from e

        if not content:
            raise ValueError(f"attachments_base64[{index}] ({filename}) is empty.")
        if len(content) > ATTACHMENT_MAX_BYTES:
            raise ValueError(
                f"attachments_base64[{index}] ({filename}) exceeds the "
                f"{ATTACHMENT_MAX_BYTES // (1024 * 1024)} MiB inline limit."
            )

        decoded.append({"filename": filename, "content": content})

    return decoded
