from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embedding
from src.vector_store import VectorStore


PDF_PATH = "data/uploads/sample.pdf"

INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.pkl"


# --------------------------------------------------
# 1. Extract PDF
# --------------------------------------------------

pages = extract_text_from_pdf(PDF_PATH)

print(f"Pages extracted: {len(pages)}")


# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

chunks = chunk_pages(pages)

print(f"Chunks created: {len(chunks)}")


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

embeddings = []

print("\nGenerating embeddings...")

for i, chunk in enumerate(chunks):

    embedding = create_embedding(
        chunk["text"]
    )

    embeddings.append(embedding)

    print(
        f"Embedded chunk {i + 1}/{len(chunks)}"
    )


# --------------------------------------------------
# 4. Create FAISS vector store
# --------------------------------------------------

dimension = len(embeddings[0])

vector_store = VectorStore(
    dimension
)

vector_store.add_embeddings(
    embeddings,
    chunks
)

print("\nFAISS index ready!")


# --------------------------------------------------
# 5. Save index
# --------------------------------------------------

vector_store.save(
    INDEX_PATH,
    METADATA_PATH
)

print("\nIndex building completed!")