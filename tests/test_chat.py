import os
import types
import builtins

import pytest

import chat as chat_mod


def test_resolve_model_prefers_deployment(monkeypatch):
    monkeypatch.setenv("OPENAI_DEPLOYMENT", "azure-deploy")
    monkeypatch.setenv("OPENAI_MODEL", "ignored-model")
    assert chat_mod.resolve_model() == "azure-deploy"


def test_resolve_model_model_fallback(monkeypatch):
    # Set to empty so load_dotenv won't override
    monkeypatch.setenv("OPENAI_DEPLOYMENT", "")
    monkeypatch.setenv("OPENAI_MODEL", "some-model")
    assert chat_mod.resolve_model() == "some-model"


def test_resolve_model_missing_raises(monkeypatch):
    # Set both to empty so load_dotenv won't override from .env
    monkeypatch.setenv("OPENAI_DEPLOYMENT", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    with pytest.raises(SystemExit) as exc:
        chat_mod.resolve_model()
    assert exc.value.code == 2


# ---- Fake client for chat_once tests (no network) ----


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _Delta:
    def __init__(self, content: str | None):
        self.content = content


class _StreamChoice:
    def __init__(self, content: str | None):
        self.delta = _Delta(content)


class _StreamEvent:
    def __init__(self, content: str | None):
        self.choices = [_StreamChoice(content)]


class _FakeCompletions:
    def __init__(self, stream_chunks: list[str] | None = None, final_text: str = "OK"):
        self._stream_chunks = stream_chunks or []
        self._final_text = final_text

    def create(self, *, model, messages, stream=False):  # noqa: D401 - match signature
        # Return iterable of events for stream=True, else a non-streaming completion
        if stream:
            def _iter():
                for chunk in self._stream_chunks:
                    yield _StreamEvent(chunk)
                # Send one "None" to simulate end with no content
                yield _StreamEvent(None)

            return _iter()
        else:
            # Echo back a predictable response
            return _FakeCompletion(self._final_text)


class _FakeChat:
    def __init__(self, stream_chunks=None, final_text="OK"):
        self.completions = _FakeCompletions(stream_chunks=stream_chunks, final_text=final_text)


class _FakeClient:
    def __init__(self, stream_chunks=None, final_text="OK"):
        self.chat = _FakeChat(stream_chunks=stream_chunks, final_text=final_text)


def test_chat_once_non_streaming():
    client = _FakeClient(final_text="NONSTREAM")
    messages: list[dict] = []
    out = chat_mod.chat_once(client, "dummy-model", messages, "hello", stream=False, console=None)  # type: ignore[arg-type]
    assert out == "NONSTREAM"
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1].get("content") == "NONSTREAM"


def test_chat_once_streaming():
    client = _FakeClient(stream_chunks=["he", "llo"])  # yields "hello"
    messages: list[dict] = []
    out = chat_mod.chat_once(client, "dummy-model", messages, "hi", stream=True, console=None)  # type: ignore[arg-type]
    assert out == "hello"
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1].get("content") == "hello"
