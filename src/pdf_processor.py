import pymupdf


def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):

        page_width = page.rect.width

        blocks = page.get_text("blocks")

        full_width_blocks = []
        left_column = []
        right_column = []

        for block in blocks:

            x0, y0, x1, y1, text, *_ = block

            text = text.strip()

            if not text:
                continue

            block_width = x1 - x0

            # Blocks spanning most of the page width
            # are usually titles, headings, etc.
            if block_width > page_width * 0.75:
                full_width_blocks.append(
                    (y0, text)
                )

            elif x0 < page_width / 2:
                left_column.append(
                    (y0, text)
                )

            else:
                right_column.append(
                    (y0, text)
                )

        # Sort blocks vertically
        full_width_blocks.sort(key=lambda x: x[0])
        left_column.sort(key=lambda x: x[0])
        right_column.sort(key=lambda x: x[0])

        # Reconstruct page in reading order:
        # full-width content → left column → right column
        ordered_blocks = (
            full_width_blocks
            + left_column
            + right_column
        )

        page_text = "\n\n".join(
            text for _, text in ordered_blocks
        )

        if page_text.strip():

            pages.append({
                "page_number": page_number + 1,
                "text": page_text
            })

    doc.close()

    return pages