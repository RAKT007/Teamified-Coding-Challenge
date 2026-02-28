from rag_core import Config, load_or_build_vectorstore

class DummyEmb:
    pass

class DummyVS:
    def save_local(self, path):
        self.saved_to = path

def test_load_or_build_vectorstore_loads_when_index_exists(tmp_path, monkeypatch):
    import rag_core
    idx_dir = tmp_path / "faiss_store"
    idx_dir.mkdir()

    monkeypatch.setattr(rag_core, "TransformersMeanPoolingEmbeddings", lambda *a, **k: DummyEmb())
    monkeypatch.setattr(rag_core.FAISS, "load_local", lambda *a, **k: "LOADED_VS")

    cfg = Config(
        pdf_path="x.pdf",
        groq_model="m",
        embed_model="e",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        embed_max_length=256,
        index_dir=str(idx_dir),
    )

    vs = load_or_build_vectorstore(cfg)
    assert vs == "LOADED_VS"

def test_load_or_build_vectorstore_builds_when_missing(tmp_path, monkeypatch):
    import rag_core
    idx_dir = tmp_path / "faiss_store"  # doesn't exist

    monkeypatch.setattr(rag_core, "TransformersMeanPoolingEmbeddings", lambda *a, **k: DummyEmb())
    monkeypatch.setattr(rag_core, "load_pdf_text_from_path", lambda p: "Hello " * 1000)
    monkeypatch.setattr(rag_core, "make_documents", lambda t, cs, co: [])
    monkeypatch.setattr(rag_core.FAISS, "from_documents", lambda docs, emb: DummyVS())

    cfg = Config(
        pdf_path="philippine_history.pdf",
        groq_model="m",
        embed_model="e",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        embed_max_length=256,
        index_dir=str(idx_dir),
    )

    vs = load_or_build_vectorstore(cfg)
    assert isinstance(vs, DummyVS)
    assert vs.saved_to == str(idx_dir)