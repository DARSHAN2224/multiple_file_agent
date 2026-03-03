import fitz
import glob

files = glob.glob(r'A:\Projects\assisto\Multi_document_Agent\temp_uploads\*.pdf')
if not files:
    print('No temp PDF found - trying to locate source file...')
    files = glob.glob(r'A:\**\MULTI_DOC_SUMMARIZATION.pdf', recursive=True)

if not files:
    print('No PDF found. Please place a test PDF at temp_uploads/test.pdf')
else:
    path = files[0]
    print(f'Testing: {path}')
    doc = fitz.open(path)
    print(f'Pages: {len(doc)}')

    page = doc.load_page(0)
    plain_text = page.get_text()
    print(f'Plain text length (page 1): {len(plain_text)}')
    print(f'Text preview: {repr(plain_text[:400])}')

    blocks = page.get_text('dict')['blocks']
    print(f'Blocks on page 1: {len(blocks)}')

    spans = []
    for b in blocks:
        if 'lines' in b:
            for line in b['lines']:
                for s in line['spans']:
                    spans.append(s)

    print(f'Spans on page 1: {len(spans)}')
    if spans:
        sizes = [s['size'] for s in spans]
        print(f'Font sizes: min={min(sizes):.1f}, max={max(sizes):.1f}')
        print(f'Median size: {sorted(sizes)[len(sizes)//2]:.1f}')
        for s in spans[:10]:
            print(f'  size={s["size"]:.1f} font={s["font"]} text={repr(s["text"][:60])}')
    else:
        print('No spans found - PDF may be image-based (scanned)!')
        print('Try using OCR (pytesseract or similar) to extract text.')
