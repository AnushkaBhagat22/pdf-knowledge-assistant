from src.pdf_processor import extract_text_from_pdf


pdf_path = "data/uploads/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print(f"Total pages extracted: {len(pages)}")

for page in pages:
    print("\n" + "=" * 60)
    print(f"PAGE {page['page_number']}")
    print("=" * 60)
    print(page["text"][:1000])