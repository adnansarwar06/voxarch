import faiss
import pickle
import numpy as np
import logging
from voxarch.utils.config import Config
from voxarch.rag.ingest import ingest_books_and_audio
from voxarch.rag.embedder import TextEmbedder
from voxarch.rag.audioembedder import AudioEmbedder

logger = logging.getLogger("voxarch.vectorstore")
logger.setLevel(logging.INFO)

class VectorStore:
    """
    Builds, saves, loads, and queries a unified FAISS index for text and audio data.
    Logs all key operations and errors.
    """
    def __init__(self, config_path=None):
        config = Config(config_path) if config_path else Config()
        self.config = config
        self.index_path = config.get("faiss.index_path", "data/vector.index")
        self.text_model_name = config.get("models.text_embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.audio_model_name = config.get("models.audio_embedding_model", "laion/clap-htsat-unfused")
        self.audio_sample_rate = config.get("audio.sample_rate", 48000)
        try:
            self.embedder = TextEmbedder(self.text_model_name)
            self.audio_embedder = AudioEmbedder(model_name=self.audio_model_name, sample_rate=self.audio_sample_rate)
            logger.info(f"VectorStore initialized with models: text='{self.text_model_name}', audio='{self.audio_model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize embedders: {e}")
            raise
        self.index = None
        self.metadata = []

    def build(self):
        """
        Ingests all content and builds the unified FAISS index with metadata.
        """
        try:
            chunks, metadata = ingest_books_and_audio(self.config)
            embeddings = []
            for chunk, meta in zip(chunks, metadata):
                try:
                    # Use audio or text embedder depending on chunk type
                    if meta.get("source_type") == "audio_clap":
                        emb = self.audio_embedder.embed(chunk)[0]
                    else:
                        emb = self.embedder.embed(chunk)[0]
                    embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Embedding failed for chunk: {meta}. Error: {e}")
                    continue
            if not embeddings:
                logger.error("No embeddings generated. Aborting index build.")
                raise RuntimeError("No embeddings generated for the dataset.")
            embeddings = np.stack(embeddings)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            self.metadata = metadata
            logger.info(f"Built FAISS index with {len(embeddings)} vectors (dim={dim}).")
        except Exception as e:
            logger.error(f"Index build failed: {e}")
            raise

    def save(self):
        """
        Saves the FAISS index and metadata to disk.
        """
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.index_path + ".meta", "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Index and metadata saved to '{self.index_path}' and '{self.index_path}.meta'.")
        except Exception as e:
            logger.error(f"Failed to save index or metadata: {e}")
            raise

    def load(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.index_path + ".meta", "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index and metadata from '{self.index_path}'.")
        except Exception as e:
            logger.error(f"Failed to load index or metadata: {e}")
            raise

    def query(self, query_text, top_k=None):
        """
        Queries the index using text; returns top-k matching chunks and their metadata.
        """
        try:
            top_k = top_k or self.config.get("search.top_k", 5)
            emb = self.embedder.embed(query_text)
            D, I = self.index.search(emb, top_k)
            results = []
            for idx, score in zip(I[0], D[0]):
                meta = self.metadata[idx].copy()
                meta["score"] = float(score)
                results.append(meta)
            logger.info(f"Text query for '{query_text[:30]}...' returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Query failed for '{query_text}': {e}")
            raise

    def query_audio(self, audio_path, top_k=None):
        """
        Queries the index using an audio file. Transcribes audio, then uses transcript for semantic search.
        """
        try:
            from voxarch.rag.audio import transcribe_audio
            whisper_model = self.config.get("audio.whisper_model", "base")
            transcript = transcribe_audio(audio_path, whisper_model=whisper_model)
            results = self.query(transcript, top_k=top_k)
            logger.info(f"Audio query from '{audio_path}' returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Audio query failed for '{audio_path}': {e}")
            raise
