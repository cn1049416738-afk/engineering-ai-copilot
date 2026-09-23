from pypdf import PdfReader


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)

    chunks = []
    page_numbers = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            page_chunks = split_text(text)

            for chunk in page_chunks:
                chunks.append(chunk)
                page_numbers.append(page_index + 1)

    return chunks, page_numbers