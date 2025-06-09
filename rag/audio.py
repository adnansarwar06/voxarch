import os
import whisper
import torchaudio
import logging
from voxarch.utils.config import Config

logger = logging.getLogger("voxarch.audio")
logger.setLevel(logging.INFO)

def transcribe_audio(path, whisper_model=None, max_len=None, return_segments=False):
    """
    Transcribes an audio file using Whisper.
    Returns the full transcript and, if requested, segment information for timestamped chunking.
    Logs success and errors.
    """
    config = Config()
    whisper_model = whisper_model or config.get("audio.whisper_model", "base")
    max_len = max_len or config.get("audio.max_audio_length_sec", None)
    try:
        model = whisper.load_model(whisper_model)
        audio = whisper.load_audio(path)
        if max_len is not None:
            audio = audio[:int(max_len * whisper.audio.SAMPLE_RATE)]
        result = model.transcribe(audio, word_timestamps=True)
        transcript = result["text"]
        segments = result.get("segments", [])
        logger.info(f"Transcribed audio: {path} (model={whisper_model}, len={len(audio)/whisper.audio.SAMPLE_RATE:.1f}s)")
        if return_segments:
            return transcript, segments
        return transcript
    except Exception as e:
        logger.error(f"Failed to transcribe audio {path} with Whisper: {e}")
        raise

def get_audio_files(audio_dir=None, supported_exts=None):
    """
    Finds all audio files in a directory matching the given extensions.
    Logs the count found.
    """
    config = Config()
    audio_dir = audio_dir or config.get("data.audio_dir", "./data/audio")
    supported_exts = supported_exts or config.get("audio.supported_extensions", [".wav", ".mp3", ".m4a", ".flac"])
    try:
        files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if os.path.splitext(f)[1].lower() in supported_exts
        ]
        logger.info(f"Found {len(files)} audio files in {audio_dir}.")
        return files
    except Exception as e:
        logger.error(f"Error listing audio files in {audio_dir}: {e}")
        return []

def chunk_audio_transcript(segments, chunk_size=None, overlap=None):
    """
    Chunks a Whisper transcript (using segments) by word count.
    Returns a list of tuples: (chunk_text, start_time, end_time).
    Logs chunking progress and errors.
    """
    config = Config()
    chunk_size = chunk_size or config.get("chunking.chunk_size", 400)
    overlap = overlap or config.get("chunking.overlap", 50)
    try:
        chunks = []
        buffer = []
        buffer_start = None
        last_end = None
        words = 0
        for seg in segments:
            seg_text = seg["text"].strip()
            seg_start = seg.get("start", None)
            seg_end = seg.get("end", None)
            seg_words = seg_text.split()
            if buffer_start is None:
                buffer_start = seg_start
            buffer.extend(seg_words)
            last_end = seg_end
            words += len(seg_words)
            # Create chunk if threshold met
            if words >= chunk_size:
                chunks.append((" ".join(buffer), buffer_start, last_end))
                if overlap > 0 and len(buffer) > overlap:
                    buffer = buffer[-overlap:]
                    words = len(buffer)
                    buffer_start = last_end - (seg_end - seg_start) * (overlap / len(seg_words))
                else:
                    buffer = []
                    words = 0
                    buffer_start = None
        if buffer:
            chunks.append((" ".join(buffer), buffer_start, last_end))
        logger.info(f"Chunked transcript into {len(chunks)} audio chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking transcript: {e}")
        raise
