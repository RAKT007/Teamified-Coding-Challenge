import os
import pytest
from rag_core import load_pdf_text_from_path

@pytest.mark.integration
def test_real_pdf_loads():
    if not os.path.exists("philippine_history.pdf"):
        pytest.skip("philippine_history.pdf not present in test environment")

    text = load_pdf_text_from_path("philippine_history.pdf")
    assert len(text) > 10000
    assert "Philippine" in text