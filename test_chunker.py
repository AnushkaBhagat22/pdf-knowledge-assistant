from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_pages


pdf_path = "data/uploads/sample.pdf"

# Step 1: Extract pages
pages = extract_text_from_pdf(pdf_path)

# Step 2: Create chunks
chunks = chunk_pages(
    pages,
    chunk_size=1000,
    overlap=200
)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:10]):
    print("\n" + "=" * 70)
    print(f"CHUNK {i + 1}")
    print(f"PAGE: {chunk['page_number']}")
    print("=" * 70)
    print(chunk["text"])