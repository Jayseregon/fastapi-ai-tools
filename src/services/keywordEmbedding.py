from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.models.models import EmbeddedKeyword, Embeddings


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Encode a list of keywords into embeddings."""
        return self.model.encode(keywords)

    def reduce_dimensions(
        self, embeddings: np.ndarray, n_components: int = 2
    ) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        if len(embeddings) == 1:
            # For single word, return a point at origin
            return np.zeros((1, 2))
        elif len(embeddings) == 2:
            # For two words, use a line
            return np.array([[0, 0], [1, 0]])
        else:
            pca = PCA(n_components=n_components)
            return pca.fit_transform(embeddings)

    def get_normalized_list(
        self, embeddings: np.ndarray, value_range: Tuple[float, float] = (0, 1)
    ) -> List:
        """Normalize the reduced embeddings within the specified range."""
        scaler = MinMaxScaler(feature_range=value_range)
        normalized = scaler.fit_transform(embeddings)
        return normalized.tolist()

    def get_embeddings(
        self, normalized_embeddings: List, keywords: List[str]
    ) -> Embeddings:
        """Map normalized embeddings to their corresponding keywords."""
        return Embeddings(
            keywords=[
                EmbeddedKeyword(word=word, x=x, y=y)
                for word, (x, y) in zip(keywords, normalized_embeddings)
            ]
        )

    def process_keywords(self, keywords: List[str]) -> Embeddings:
        """Generate embeddings from keywords and structure them in the Embeddings schema."""
        if not keywords:
            return Embeddings(keywords=[])

        unique_keywords = list(dict.fromkeys(keywords))  # Preserve order
        embeddings = self.create_embeddings(unique_keywords)
        reduced = self.reduce_dimensions(embeddings)
        normalized = self.get_normalized_list(reduced)

        # Create mapping for duplicate keywords
        embedding_map = {
            word: coords for word, coords in zip(unique_keywords, normalized)
        }

        # Return embeddings maintaining original order and duplicates
        return Embeddings(
            keywords=[
                EmbeddedKeyword(
                    word=word, x=embedding_map[word][0], y=embedding_map[word][1]
                )
                for word in keywords
            ]
        )
