from rag_core import make_documents

def test_make_documents_creates_docs_with_chunk_ids():
    text = "A " * 5000
    docs = make_documents(text, chunk_size=200, chunk_overlap=50)

    assert len(docs) > 5
    assert docs[0].metadata["chunk_id"] == 0
    assert docs[1].metadata["chunk_id"] == 1
    assert all(len(d.page_content) > 0 for d in docs)