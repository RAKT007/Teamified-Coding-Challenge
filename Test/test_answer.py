from types import SimpleNamespace
from langchain_core.documents import Document
from rag_core import answer_with_groq

class CapturingLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return SimpleNamespace(content="FINAL_ANSWER")

def test_answer_with_groq_builds_prompt(monkeypatch):
    import rag_core
    fake = CapturingLLM()
    monkeypatch.setattr(rag_core, "ChatGroq", lambda **kwargs: fake)

    docs = [
        Document(page_content="Chunk text 1", metadata={"chunk_id": 10}),
        Document(page_content="Chunk text 2", metadata={"chunk_id": 11}),
    ]
    out = answer_with_groq("key", "model", "My question", docs, "summary")

    assert out == "FINAL_ANSWER"
    assert "Answer ONLY using the provided context" in fake.last_prompt
    assert "Intent: summary" in fake.last_prompt
    assert "[chunk_id=10]" in fake.last_prompt
    assert "Chunk text 1" in fake.last_prompt
    assert "My question" in fake.last_prompt