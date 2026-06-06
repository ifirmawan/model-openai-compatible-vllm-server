"""Unit tests for vllm_inference.py.

These tests run locally and do NOT require a Modal deployment or GPU.
They verify configuration constants and the SSE parsing logic in _send_request.
"""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test (no Modal infra is initialised at import time)
# ---------------------------------------------------------------------------
import vllm_inference as sut


# ---------------------------------------------------------------------------
# Helper: async iterator from a plain list of byte lines
# ---------------------------------------------------------------------------
class AsyncIteratorMock:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def make_resp_mock(lines: list[bytes]):
    """Build a mock aiohttp response with an async-iterable .content."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = AsyncIteratorMock(lines)
    return resp


def make_session_mock(resp):
    """Wrap resp in a session whose .post() is an async context manager."""
    session = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    session.post.return_value = cm
    return session


def make_chunk(delta: dict) -> bytes:
    chunk = {"object": "chat.completion.chunk", "choices": [{"delta": delta}]}
    return f"data: {json.dumps(chunk)}\n".encode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestConstants(unittest.TestCase):
    """Sanity-check the key constants so accidental edits are caught."""

    def test_model_name(self):
        assert sut.MODEL_NAME == "google/gemma-4-26B-A4B-it"

    def test_model_revision_is_pinned(self):
        assert len(sut.MODEL_REVISION) == 40
        assert all(c in "0123456789abcdef" for c in sut.MODEL_REVISION)

    def test_speculative_model_name(self):
        assert "assistant" in sut.SPECULATIVE_MODEL_NAME

    def test_vllm_port(self):
        assert sut.VLLM_PORT == 8000

    def test_n_gpu(self):
        assert sut.N_GPU >= 1

    def test_fast_boot_is_bool(self):
        assert isinstance(sut.FAST_BOOT, bool)


class TestSendRequestParsing(unittest.IsolatedAsyncioTestCase):
    """Test the SSE chunk-parsing logic inside _send_request."""

    async def test_content_delta_printed(self):
        """Content tokens in delta.content should be printed."""
        resp = make_resp_mock([make_chunk({"content": "Hello"}), b"data: [DONE]\n"])
        session = make_session_mock(resp)

        with patch("builtins.print") as mock_print:
            await sut._send_request(session, "llm", [])

        printed = "".join(str(a) for call in mock_print.call_args_list for a in call.args)
        assert "Hello" in printed

    async def test_done_sentinel_skipped(self):
        """The [DONE] SSE line must not raise."""
        resp = make_resp_mock([b"data: [DONE]\n"])
        session = make_session_mock(resp)
        await sut._send_request(session, "llm", [])  # should not raise

    async def test_reasoning_delta_printed(self):
        """Tokens in delta.reasoning should also be printed."""
        resp = make_resp_mock([
            make_chunk({"reasoning": "<think>step</think>"}),
            b"data: [DONE]\n",
        ])
        session = make_session_mock(resp)

        with patch("builtins.print") as mock_print:
            await sut._send_request(session, "llm", [])

        printed = "".join(str(a) for call in mock_print.call_args_list for a in call.args)
        assert "<think>" in printed

    async def test_wrong_object_type_raises(self):
        """A chunk with the wrong 'object' field should trigger an AssertionError."""
        bad_chunk = json.dumps({
            "object": "chat.completion",  # not a chunk!
            "choices": [{"delta": {"content": "oops"}}],
        })
        resp = make_resp_mock([f"data: {bad_chunk}\n".encode()])
        session = make_session_mock(resp)

        with pytest.raises(AssertionError):
            await sut._send_request(session, "llm", [])
