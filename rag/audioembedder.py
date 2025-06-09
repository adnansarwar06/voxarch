import logging
import soundfile as sf
import numpy as np
import openl3
from voxarch.utils.config import Config

logger = logging.getLogger("voxarch.audioembedder")
logger.setLevel(logging.INFO)

class AudioEmbedder:
    """
    Audio embedder using OpenL3 for deep audio embeddings (semantic rich).
    """
    def __init__(self, input_repr=None, content_type=None, embedding_size=None):
        config = Config()
        self.input_repr = input_repr or config.get("audio.openl3_input_repr", "mel128")
        self.content_type = content_type or config.get("audio.openl3_content_type", "music")
        self.embedding_size = embedding_size or config.get("audio.openl3_embedding_size", 512)
        logger.info(f"Initialized OpenL3 embedder: {self.input_repr}/{self.content_type}/emb{self.embedding_size}")

    def embed(self, audio_paths):
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        embeddings = []
        for path in audio_paths:
            try:
                audio, sr = sf.read(path, always_2d=True)
                emb, ts = openl3.get_audio_embedding(
                    audio,
                    sr,
                    input_repr=self.input_repr,
                    content_type=self.content_type,
                    embedding_size=self.embedding_size
                )
                # Pool over frames: mean
                emb_mean = np.mean(emb, axis=0)
                embeddings.append(emb_mean)
                logger.info(f"Embedded audio via OpenL3: {path}")
            except Exception as e:
                logger.error(f"Failed to embed audio {path} via OpenL3: {e}")
        if not embeddings:
            logger.warning("No audio embeddings produced.")
            return np.zeros((1, self.embedding_size))
        return np.stack(embeddings)
