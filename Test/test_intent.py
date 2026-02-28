from types import SimpleNamespace
from rag_core import detect_intent_with_groq

class FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self._content)

def test_detect_intent_valid_json(monkeypatch):
    import rag_core
    monkeypatch.setattr(rag_core, "ChatGroq", lambda **kwargs: FakeLLM('{"intent":"person_bio"}'))

    out = detect_intent_with_groq("key", "model", "Who is Rizal?")
    assert out == "person_bio"

def test_detect_intent_invalid_json_falls_back(monkeypatch):
    import rag_core
    monkeypatch.setattr(rag_core, "ChatGroq", lambda **kwargs: FakeLLM("nonsense"))

    out = detect_intent_with_groq("key", "model", "When was EDSA?")
    assert out == "other"

def test_detect_intent_unknown_label_falls_back(monkeypatch):
    import rag_core
    monkeypatch.setattr(rag_core, "ChatGroq", lambda **kwargs: FakeLLM('{"intent":"nope"}'))

    out = detect_intent_with_groq("key", "model", "test")
    assert out == "other"