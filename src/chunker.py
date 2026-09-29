import re


def split_into_sentences(text):
    """
    Split text into approximate sentences.
    """
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_pages(pages, chunk_size=1200, overlap=200):
    """
    Create sentence-aware chunks while preserving page numbers.
    """

    chunks = []

    for page in pages:
        page_number = page["page_number"]

        sentences = split_into_sentences(page["text"])

        current_chunk = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            # If adding the sentence exceeds the chunk size,
            # save the current chunk first.
            if (
                current_length + sentence_length > chunk_size
                and current_chunk
            ):
                chunk_text = " ".join(current_chunk)

                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number
                })

                # Keep the last few sentences for overlap
                overlap_chunk = []
                overlap_length = 0

                for previous_sentence in reversed(current_chunk):

                    if overlap_length + len(previous_sentence) > overlap:
                        break

                    overlap_chunk.insert(0, previous_sentence)
                    overlap_length += len(previous_sentence)

                current_chunk = overlap_chunk
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Save remaining text from the page
        if current_chunk:

            chunk_text = " ".join(current_chunk)

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number
                })

    return chunks