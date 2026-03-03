import re
import fitz  # PyMuPDF
import docx
import pandas as pd
from typing import Generator, List
from backend.models.schemas import SourceChunk
import tiktoken  # Better estimation of tokenizer

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def estimate_tokens(text: str) -> int:
    # A rough estimation using tiktoken standard cl100k_base to avoid loading massive llama tokenizer for every chunk
    # Usually 1 token ~= 4 chars in english
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

class DocumentProcessor:
    """Handles parsing and section-aware chunking of PDF, DOCX, and TXT files."""
    
    @staticmethod
    def _ocr_page(page, page_num: int, source_name: str, current_section: str) -> Generator:
        """Render a page to image and extract text via Tesseract OCR."""
        if not OCR_AVAILABLE:
            return
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(img, lang="eng")
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', ocr_text) if p.strip()]
        for para in paragraphs:
            if len(para) < 10:
                continue
            yield SourceChunk(
                source_file=source_name,
                page_number=page_num + 1,
                section_title=current_section,
                text=para,
                token_count=estimate_tokens(para)
            )

    @staticmethod
    def process_pdf(file_path: str, source_name: str) -> Generator[SourceChunk, None, None]:
        with fitz.open(file_path) as doc:
            # First pass: collect all font sizes to compute a robust median threshold
            all_sizes = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                for block in page.get_text("dict")["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    all_sizes.append(span["size"])

            if all_sizes:
                all_sizes.sort()
                median_size = all_sizes[len(all_sizes) // 2]
                heading_threshold = median_size * 1.2
            else:
                heading_threshold = 14.0

            current_section = "General"
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text_parts = []
                found_any_text = False

                for block in page.get_text("dict")["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue
                                found_any_text = True
                                is_bold = "bold" in span["font"].lower()
                                is_large = span["size"] >= heading_threshold
                                if is_large and is_bold and len(text) < 120:
                                    if page_text_parts:
                                        full_text = " ".join(page_text_parts).strip()
                                        if full_text:
                                            yield SourceChunk(
                                                source_file=source_name,
                                                page_number=page_num + 1,
                                                section_title=current_section,
                                                text=full_text,
                                                token_count=estimate_tokens(full_text)
                                            )
                                        page_text_parts = []
                                    current_section = text
                                else:
                                    page_text_parts.append(text)

                if page_text_parts:
                    full_text = " ".join(page_text_parts).strip()
                    if full_text:
                        yield SourceChunk(
                            source_file=source_name,
                            page_number=page_num + 1,
                            section_title=current_section,
                            text=full_text,
                            token_count=estimate_tokens(full_text)
                        )

                # OCR fallback: if no text spans were found on this page, run Tesseract
                if not found_any_text and OCR_AVAILABLE:
                    yield from DocumentProcessor._ocr_page(page, page_num, source_name, current_section)
    
    @staticmethod
    def process_docx(file_path: str, source_name: str) -> Generator[SourceChunk, None, None]:
        doc = docx.Document(file_path)
        current_section = "General"

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Check if paragraph is a heading by style
            if para.style.name.startswith('Heading'):
                current_section = text
            else:
                chunk = SourceChunk(
                    source_file=source_name,
                    page_number=1, # DOCX doesn't easily expose pages
                    section_title=current_section,
                    text=text,
                    token_count=estimate_tokens(text)
                )
                yield chunk

    @staticmethod
    def process_txt(file_path: str, source_name: str) -> Generator[SourceChunk, None, None]:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_section = "General"
            for line_no, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue

                # Basic markdown header detection
                if re.match(r'^#{1,6}\s', text):
                    current_section = text.lstrip('#').strip()
                else:
                    chunk = SourceChunk(
                        source_file=source_name,
                        page_number=1, # TXT has no pages
                        section_title=current_section,
                        text=text,
                        token_count=estimate_tokens(text)
                    )
                    yield chunk

    @staticmethod
    def process_csv(file_path: str, source_name: str) -> Generator[SourceChunk, None, None]:
        """Reads a CSV and yields row-batches as chunks."""
        df = pd.read_csv(file_path, dtype=str).fillna("")
        columns = list(df.columns)
        batch_rows = []
        batch_tokens = 0
        BATCH_LIMIT = 300

        for idx, row in df.iterrows():
            row_text = "  |  ".join(f"{col}: {val}" for col, val in zip(columns, row) if val.strip())
            if not row_text.strip():
                continue
            tokens = estimate_tokens(row_text)
            if batch_tokens + tokens > BATCH_LIMIT and batch_rows:
                chunk_text = "\n".join(batch_rows)
                yield SourceChunk(
                    source_file=source_name,
                    page_number=1,
                    section_title=f"Columns: {', '.join(columns)}",
                    text=chunk_text,
                    token_count=estimate_tokens(chunk_text)
                )
                batch_rows = []
                batch_tokens = 0
            batch_rows.append(row_text)
            batch_tokens += tokens

        if batch_rows:
            chunk_text = "\n".join(batch_rows)
            yield SourceChunk(
                source_file=source_name,
                page_number=1,
                section_title=f"Columns: {', '.join(columns)}",
                text=chunk_text,
                token_count=estimate_tokens(chunk_text)
            )

    @staticmethod
    def process_excel(file_path: str, source_name: str) -> Generator[SourceChunk, None, None]:
        """Reads each Excel sheet as a section and yields row-batches as chunks."""
        with pd.ExcelFile(file_path) as xls:
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name, dtype=str).fillna("")
                columns = list(df.columns)
                batch_rows = []
                batch_tokens = 0
                BATCH_LIMIT = 300

                for idx, row in df.iterrows():
                    row_text = "  |  ".join(f"{col}: {val}" for col, val in zip(columns, row) if str(val).strip())
                    if not row_text.strip():
                        continue
                    tokens = estimate_tokens(row_text)
                    if batch_tokens + tokens > BATCH_LIMIT and batch_rows:
                        chunk_text = "\n".join(batch_rows)
                        yield SourceChunk(
                            source_file=source_name,
                            page_number=1,
                            section_title=f"Sheet: {sheet_name} | Columns: {', '.join(columns)}",
                            text=chunk_text,
                            token_count=estimate_tokens(chunk_text)
                        )
                        batch_rows = []
                        batch_tokens = 0
                    batch_rows.append(row_text)
                    batch_tokens += tokens

                if batch_rows:
                    chunk_text = "\n".join(batch_rows)
                    yield SourceChunk(
                        source_file=source_name,
                        page_number=1,
                        section_title=f"Sheet: {sheet_name} | Columns: {', '.join(columns)}",
                        text=chunk_text,
                        token_count=estimate_tokens(chunk_text)
                    )

    @classmethod
    def process_file(cls, file_path: str, source_name: str) -> List[SourceChunk]:
        """Entry point that routes to the correct specific processor."""
        ext = file_path.lower().split('.')[-1]
        # CSV and Excel are returned directly — rows are naturally atomic
        if ext == 'csv':
            return list(cls.process_csv(file_path, source_name))
        elif ext in ('xlsx', 'xls'):
            return list(cls.process_excel(file_path, source_name))

        raw_chunks = []
        if ext == 'pdf':
            raw_chunks = list(cls.process_pdf(file_path, source_name))
        elif ext == 'docx':
            raw_chunks = list(cls.process_docx(file_path, source_name))
        elif ext == 'txt':
            raw_chunks = list(cls.process_txt(file_path, source_name))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Agglomerate small chunks into larger context-rich blocks respecting boundaries
        merged_chunks = []
        current_merged = None
        
        for c in raw_chunks:
            if current_merged is None:
                current_merged = c
            else:
                if (current_merged.section_title == c.section_title and 
                    current_merged.page_number == c.page_number and
                    current_merged.token_count + c.token_count < 400):
                    current_merged.text += " " + c.text
                    current_merged.token_count += c.token_count
                else:
                    merged_chunks.append(current_merged)
                    current_merged = c
                    
        if current_merged:
            merged_chunks.append(current_merged)
            
        return merged_chunks
