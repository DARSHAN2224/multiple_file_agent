import pytest
import os
from backend.core.document_processor import DocumentProcessor

def test_process_txt_file(tmp_path):
    # Create a dummy txt file
    dummy_txt = tmp_path / "sample.txt"
    dummy_txt.write_text("# Main Header\nThis is paragraph one.\n# Sec 2\nThis is paragraph two.", encoding="utf-8")
    
    chunks = DocumentProcessor.process_file(str(dummy_txt), "sample.txt")
    
    assert len(chunks) == 2
    assert chunks[0].section_title == "Main Header"
    assert "paragraph one" in chunks[0].text
    
    assert chunks[1].section_title == "Sec 2"
    assert "paragraph two" in chunks[1].text
    assert chunks[1].page_number == 1 # Text files default to page 1

def test_process_empty_txt(tmp_path):
    dummy_txt = tmp_path / "empty.txt"
    dummy_txt.write_text("")
    
    chunks = DocumentProcessor.process_file(str(dummy_txt), "empty.txt")
    assert len(chunks) == 0
