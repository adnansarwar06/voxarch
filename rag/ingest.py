import os
import re
from tqdm import tqdm
import logging
from voxarch.utils.config import Config
from voxarch.rag.audio import transcribe_audio, get_audio_files, chunk_audio_transcript
import pdfplumber
import pytesseract
from PIL import Image
from ebooklib import epub
from bs4 import BeautifulSoup
import docx

logger = logging.getLogger("voxarch.ingest")
logger.setLevel(logging.INFO)

def extract_text_txt(path):
    """
    Loads raw text from a .txt file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Loaded .txt file: {path}")
        return text
    except Exception as e:
        logger.error(f"Failed to load .txt file {path}: {e}")
        raise

def extract_text_pdf(path, use_ocr=False, ocr_languages=None):
    """
    Extracts text from a PDF file, using OCR on pages with no extractable text if enabled.
    """
    text = ""
    ocr_languages = ocr_languages or ['eng']
    ocr_lang_str = '+'.join(ocr_languages)
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if not page_text and use_ocr:
                    image = page.to_image(resolution=300).original
                    page_text = pytesseract.image_to_string(image, lang=ocr_lang_str)
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Extracted text from PDF: {path}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {path}: {e}")
        raise

def extract_text_docx(path):
    """
    Loads text content from a .docx file.
    """
    try:
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        logger.info(f"Loaded .docx file: {path}")
        return text
    except Exception as e:
        logger.error(f"Failed to load .docx file {path}: {e}")
        raise

def extract_sections_txt(text, regex, exclude_headings=None):
    """
    Splits text into sections using a heading regex. Excludes specified headings.
    """
    lines = text.splitlines()
    sections = []
    buffer = []
    current_section = "Introduction"
    for line in lines:
        if re.match(regex, line.strip(), re.IGNORECASE):
            if buffer and (not exclude_headings or current_section not in exclude_headings):
                sections.append((current_section, "\n".join(buffer)))
            current_section = line.strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer and (not exclude_headings or current_section not in exclude_headings):
        sections.append((current_section, "\n".join(buffer)))
    return sections

def extract_sections_epub(path):
    """
    Extracts sections from an EPUB file by heading tags (h1–h4).
    """
    try:
        book = epub.read_epub(path)
        sections = []
        for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                section_title = heading.get_text().strip()
                section_text = ""
                sib = heading.find_next_sibling()
                while sib and sib.name not in ['h1', 'h2', 'h3', 'h4']:
                    section_text += sib.get_text(separator=' ', strip=True) + " "
                    sib = sib.find_next_sibling()
                if section_text:
                    sections.append((section_title, section_text.strip()))
        logger.info(f"Extracted {len(sections)} sections from EPUB: {path}")
        return sections
    except Exception as e:
        logger.error(f"Failed to extract sections from EPUB {path}: {e}")
        raise

def chunk_text(text, chunk_size=None, overlap=None, method=None, min_chunk_words=None):
    """
    Chunks text by word count or paragraph, with overlap.
    """
    config = Config()
    chunk_size = chunk_size or config.get("chunking.chunk_size", 400)
    overlap = overlap or config.get("chunking.overlap", 50)
    method = method or config.get("chunking.method", "words")
    min_chunk_words = min_chunk_words or config.get("parsing.min_chunk_words", 50)

    if method == "paragraphs":
        paras = [p for p in text.split('\n\n') if p.strip()]
        chunks = []
        i = 0
        while i < len(paras):
            chunk = "\n\n".join(paras[i:i+chunk_size])
            if len(chunk.split()) >= min_chunk_words:
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
    else:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            if len(chunk) >= min_chunk_words:
                chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        return chunks

def clean_chunk(chunk):
    """
    Cleans whitespace from a chunk.
    """
    return re.sub(r'\s+', ' ', chunk).strip()

def deduplicate_chunks(chunks):
    """
    Removes duplicate chunks (by hash).
    """
    seen = set()
    unique = []
    for c in chunks:
        key = hash(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def ingest_books_and_audio(config=None):
    """
    Ingests and processes all text and audio files from configured data folders.
    Handles parsing, sectioning, chunking, metadata creation, and prepares all content for embedding.
    Returns a tuple: (list of chunks, list of metadata dicts)
    Logs each step and handles errors gracefully.
    """
    config = config or Config()
    text_dir = config.get("data.text_dir", "data/text")
    exts = config.get("parsing.supported_extensions", [".txt", ".pdf", ".epub", ".docx"])
    section_regex = config.get("parsing.section_heading_regex", "^Chapter|^Section")
    min_section_words = config.get("parsing.min_section_words", 100)
    min_chunk_words = config.get("parsing.min_chunk_words", 50)
    exclude_headings = config.get("parsing.exclude_headings", [])
    chunk_size = config.get("chunking.chunk_size", 400)
    overlap = config.get("chunking.overlap", 50)
    chunk_method = config.get("chunking.method", "words")
    dedup = config.get("parsing.deduplicate_chunks", True)
    use_ocr_on_pdf = config.get("parsing.use_ocr_on_pdf", True)
    ocr_languages = config.get("parsing.ocr_languages", ["eng"])
    audio_dir = config.get("data.audio_dir", "./data/audio")
    audio_exts = config.get("audio.supported_extensions", [".wav", ".mp3", ".m4a", ".flac"])
    whisper_model = config.get("audio.whisper_model", "base")
    max_audio_length = config.get("audio.max_audio_length_sec", 600)
    embed_method = config.get("audio.embed_method", "both")
    clap_model_name = config.get("audio.clap_model", "laion/clap-htsat-unfused")
    clap_sample_rate = config.get("audio.sample_rate", 48000)

    files = [os.path.join(text_dir, f) for f in os.listdir(text_dir) if os.path.splitext(f)[1] in exts]
    files.sort()
    all_chunks = []
    all_metadata = []

    # --- Text file ingestion ---
    for file in tqdm(files, desc="Ingesting books"):
        ext = os.path.splitext(file)[1].lower()
        book_title = os.path.splitext(os.path.basename(file))[0]
        try:
            # Choose extraction function by file type
            if ext == ".txt":
                text = extract_text_txt(file)
                sections = extract_sections_txt(text, section_regex, exclude_headings)
            elif ext == ".pdf":
                text = extract_text_pdf(file, use_ocr=use_ocr_on_pdf, ocr_languages=ocr_languages)
                sections = extract_sections_txt(text, section_regex, exclude_headings)
            elif ext == ".epub":
                sections = extract_sections_epub(file)
            elif ext == ".docx":
                text = extract_text_docx(file)
                sections = extract_sections_txt(text, section_regex, exclude_headings)
            else:
                logger.warning(f"Unsupported file extension for {file}. Skipping.")
                continue
            if not sections:
                sections = [(book_title, text if ext != ".epub" else "")]
            for idx, (section_title, section_text) in enumerate(sections):
                if not section_text or len(section_text.split()) < min_section_words:
                    continue
                # Chunk section, deduplicate if requested
                chunks = chunk_text(section_text, chunk_size, overlap, chunk_method, min_chunk_words)
                if dedup:
                    chunks = deduplicate_chunks(chunks)
                for j, chunk in enumerate(chunks):
                    chunk = clean_chunk(chunk)
                    metadata = {
                        "source_type": "book",
                        "book_title": book_title,
                        "filename": os.path.basename(file),
                        "section": section_title,
                        "section_index": idx,
                        "chunk_index": j,
                        "text": chunk
                    }
                    all_chunks.append(chunk)
                    all_metadata.append(metadata)
            logger.info(f"Ingested {len(all_chunks)} text chunks from {file}.")
        except Exception as e:
            logger.error(f"Failed to ingest {file}: {e}")

    # --- Audio file ingestion ---
    if os.path.isdir(audio_dir):
        audio_files = get_audio_files(audio_dir, audio_exts)
        for path in tqdm(audio_files, desc="Ingesting audio"):
            base = os.path.splitext(os.path.basename(path))[0]
            try:
                transcript, segments = transcribe_audio(
                    path,
                    whisper_model=whisper_model,
                    max_len=max_audio_length,
                    return_segments=True
                )
                transcript_chunks = chunk_audio_transcript(segments, chunk_size, overlap)
                for j, (chunk, start_sec, end_sec) in enumerate(transcript_chunks):
                    metadata = {
                        "source_type": "audio",
                        "filename": os.path.basename(path),
                        "chunk_index": j,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "text": chunk,
                        "audio_path": path
                    }
                    all_chunks.append(chunk)
                    all_metadata.append(metadata)
                logger.info(f"Ingested {len(transcript_chunks)} audio transcript chunks from {path}.")
                # Add CLAP audio embedding for this file (reference by path, not text)
                if embed_method in ("audio", "both"):
                    clap_metadata = {
                        "source_type": "audio_clap",
                        "filename": os.path.basename(path),
                        "chunk_index": 0,
                        "start_time": 0,
                        "end_time": None,
                        "text": None,
                        "audio_path": path
                    }
                    all_chunks.append(path)
                    all_metadata.append(clap_metadata)
            except Exception as e:
                logger.error(f"Failed to ingest audio {path}: {e}")
    else:
        logger.warning(f"Audio directory {audio_dir} does not exist.")

    logger.info(f"Ingestion complete: {len(all_chunks)} total chunks (text+audio).")
    return all_chunks, all_metadata
