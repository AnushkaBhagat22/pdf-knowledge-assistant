from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embedding
from src.vector_store import VectorStore


# -----------------------------------------
# 1. Extract PDF text
# -----------------------------------------

pdf_path = "data/uploads/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print(f"Pages extracted: {len(pages)}")


# -----------------------------------------
# 2. Create chunks
# -----------------------------------------

chunks = chunk_pages(
    pages,
    chunk_size=1200,
    overlap=200
)

print(f"Chunks created: {len(chunks)}")


# -----------------------------------------
# 3. Generate embeddings
# -----------------------------------------

print("\nGenerating embeddings...")

embeddings = []

for i, chunk in enumerate(chunks):

    embedding = create_embedding(
        chunk["text"]
    )

    embeddings.append(embedding)

    print(
        f"Embedded chunk {i + 1}/{len(chunks)}"
    )


# -----------------------------------------
# 4. Create FAISS vector store
# -----------------------------------------

dimension = len(embeddings[0])

vector_store = VectorStore(
    dimension
)


# -----------------------------------------
# 5. Store embeddings + metadata
# -----------------------------------------

vector_store.add_embeddings(
    embeddings,
    chunks
)

print("\nFAISS index created successfully!")


# -----------------------------------------
# 6. Ask a question
# -----------------------------------------

question = (
    "Why is LSTM useful for "
    "human activity recognition?"
)

print("\nQuestion:")
print(question)


# -----------------------------------------
# 7. Embed the question
# -----------------------------------------

query_embedding = create_embedding(
    question
)


# -----------------------------------------
# 8. Search FAISS
# -----------------------------------------

results = vector_store.search(
    query_embedding,
    top_k=5
)


# -----------------------------------------
# 9. Display results
# -----------------------------------------

print("\nTop relevant chunks:\n")

for i, result in enumerate(results):

    print("=" * 70)

    print(f"RESULT {i + 1}")
    print(f"PAGE: {result['page_number']}")
    print(f"DISTANCE: {result['distance']:.4f}")

    print("\nTEXT:")
    print(result["text"][:1000])