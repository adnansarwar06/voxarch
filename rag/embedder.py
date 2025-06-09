from sentence_transformers import SentenceTransformer
from voxarch.utils.config import Config

class TextEmbedder:
    """
    Wraps a SentenceTransformer model for efficient text embedding.
    """
    def __init__(self, model_name=None):
        """
        Initializes the embedder with a given model name, using config if not provided.
        """
        config = Config()
        self.model_name = model_name or config.get("models.text_embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts):
        """
        Embeds a list of texts and returns their vector representations.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
