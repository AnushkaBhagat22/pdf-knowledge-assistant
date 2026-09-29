from src.embeddings import create_embedding


text = """
WiFi Channel State Information can be used
for human activity recognition.
"""

embedding = create_embedding(text)

print("Embedding created successfully!")
print("Embedding dimension:", len(embedding))
print("First 10 values:", embedding[:10])