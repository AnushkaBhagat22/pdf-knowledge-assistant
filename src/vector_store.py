import os
import pickle

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.metadata = []


    # ========================================================
    # ADD EMBEDDINGS
    # ========================================================

    def add_embeddings(
        self,
        embeddings,
        metadata
    ):

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension does not match "
                "the vector store dimension."
            )

        self.index.add(vectors)

        self.metadata.extend(
            metadata
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        actual_k = min(
            top_k,
            self.index.ntotal
        )

        distances, indices = (
            self.index.search(
                query_vector,
                actual_k
            )
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index == -1:
                continue

            metadata = self.metadata[
                index
            ]

            results.append(
                {
                    "text": metadata["text"],
                    "page_number": metadata[
                        "page_number"
                    ],
                    "source": metadata.get(
                        "source",
                        "Unknown document"
                    ),
                    "distance": float(
                        distance
                    )
                }
            )

        return results


    # ========================================================
    # SAVE INDEX
    # ========================================================

    def save(
        self,
        index_path,
        metadata_path
    ):

        index_directory = os.path.dirname(
            index_path
        )

        metadata_directory = os.path.dirname(
            metadata_path
        )

        if index_directory:
            os.makedirs(
                index_directory,
                exist_ok=True
            )

        if metadata_directory:
            os.makedirs(
                metadata_directory,
                exist_ok=True
            )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            metadata_path,
            "wb"
        ) as file:

            pickle.dump(
                self.metadata,
                file
            )


    # ========================================================
    # LOAD INDEX
    # ========================================================

    @classmethod
    def load(
        cls,
        index_path,
        metadata_path
    ):

        index = faiss.read_index(
            index_path
        )

        with open(
            metadata_path,
            "rb"
        ) as file:

            metadata = pickle.load(
                file
            )

        store = cls(
            index.d
        )

        store.index = index

        store.metadata = metadata

        return store