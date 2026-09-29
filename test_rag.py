from src.embeddings import create_embedding
from src.vector_store import VectorStore
from src.rag import generate_answer


INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.pkl"


# --------------------------------------------------
# 1. Load existing FAISS index
# --------------------------------------------------

print("Loading FAISS index...\n")

vector_store = VectorStore.load(
    INDEX_PATH,
    METADATA_PATH
)


# --------------------------------------------------
# 2. Ask question
# --------------------------------------------------

question = "Who invented the internet?"

print("\nQuestion:")
print(question)


# --------------------------------------------------
# 3. Create embedding for question
# --------------------------------------------------

print("\nGenerating query embedding...")

query_embedding = create_embedding(
    question
)


# --------------------------------------------------
# 4. Retrieve relevant chunks
# --------------------------------------------------

print("Searching FAISS index...")

retrieved_chunks = vector_store.search(
    query_embedding,
    top_k=5
)


# --------------------------------------------------
# 5. Generate answer
# --------------------------------------------------

print("\nGenerating answer...")

answer = generate_answer(
    question,
    retrieved_chunks
)


# --------------------------------------------------
# 6. Display answer
# --------------------------------------------------

print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)

print(answer)


# --------------------------------------------------
# 7. Display sources
# --------------------------------------------------

print("\n" + "=" * 70)
print("SOURCES")
print("=" * 70)

for chunk in retrieved_chunks:

    print(
        f"- Page {chunk['page_number']} "
        f"(distance: {chunk['distance']:.4f})"
    )