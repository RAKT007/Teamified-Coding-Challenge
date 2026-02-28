import fitz
import pytest
from rag_core import load_pdf_text_from_path

def test_load_pdf_text_from_path_reads_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF")
    doc.save(str(pdf_path))
    doc.close()

    text = load_pdf_text_from_path(str(pdf_path))
    assert "Hello PDF" in text

def test_load_pdf_text_from_path_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf_text_from_path(str(tmp_path / "missing.pdf"))