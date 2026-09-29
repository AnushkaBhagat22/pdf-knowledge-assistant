import json
import os
from pathlib import Path

from src.embeddings import create_embedding
from src.vector_store import VectorStore

INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.pkl"
QUESTIONS_PATH = "evaluation_questions.json"
TOP_K = 5


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("ERROR: FAISS index not found.")
        print("Open the Streamlit app and click 'Process PDFs' first.")
        return

    vector_store = VectorStore.load(INDEX_PATH, METADATA_PATH)
    questions = load_questions()

    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Questions: {len(questions)}")
    print(f"Top-k: {TOP_K}")
    print()

    retrieval_hits = []
    cross_document_hits = []

    for item in questions:
        print("-" * 80)
        print(f"Q{item['id']} [{item['type']}]")
        print(item["question"])

        query_embedding = create_embedding(item["question"])
        results = vector_store.search(query_embedding, top_k=TOP_K)

        retrieved_sources = []
        for result in results:
            source = result.get("source", "Unknown document")
            page = result.get("page_number", "?")
            if source not in retrieved_sources:
                retrieved_sources.append(source)
            print(f"  • {source} — Page {page} — distance={result['distance']:.4f}")

        expected = set(item["expected_sources"])

        if item["type"] == "unanswerable":
            print("  → Manual check: answer should explicitly say the information is not supported.")
            continue

        if item["type"] == "cross_document":
            hit = expected.issubset(set(retrieved_sources))
            cross_document_hits.append(hit)
            print(f"  → Both expected documents retrieved: {'PASS' if hit else 'CHECK'}")
        else:
            hit = bool(expected.intersection(retrieved_sources))
            retrieval_hits.append(hit)
            print(f"  → Expected source retrieved: {'PASS' if hit else 'CHECK'}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if retrieval_hits:
        rate = sum(retrieval_hits) / len(retrieval_hits) * 100
        print(f"Single-document retrieval hit rate: {rate:.1f}%")

    if cross_document_hits:
        rate = sum(cross_document_hits) / len(cross_document_hits) * 100
        print(f"Cross-document retrieval hit rate: {rate:.1f}%")

    print()
    print("Important: this script evaluates RETRIEVAL, not factual correctness of")
    print("the generated answer. For Q10, manually verify that the assistant refuses")
    print("to invent an exact learning rate if the uploaded documents do not provide it.")
    print()
    print("If a source appears as 'Unknown document', update VectorStore.search()")
    print("so it preserves all metadata fields, including 'source'.")


if __name__ == "__main__":
    main()
