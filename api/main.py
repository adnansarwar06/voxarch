import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from voxarch.rag.langchain_retriever import VoxarchRetriever
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from voxarch.utils.config import Config
import tempfile
import shutil

# Set up application-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("voxarch.api.main")

app = FastAPI()

def get_llm(config):
    """
    Loads the HuggingFace LLM pipeline using config.
    Logs any errors.
    """
    try:
        model_name = config.get("models.llm", "mistralai/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        logger.info(f"Loaded HuggingFace LLM: {model_name}")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise

def get_qa_chain(mode, top_k=5):
    """
    Creates the RetrievalQA chain for given mode ('text' or 'audio').
    """
    config = Config()
    llm = get_llm(config)
    retriever = VoxarchRetriever(mode=mode, top_k=top_k)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    logger.info(f"QA chain created for mode='{mode}', top_k={top_k}")
    return qa

@app.post("/query")
async def query_text(query: str = Form(...), top_k: int = Form(5)):
    """
    Answer a text question with evidence.
    """
    try:
        qa = get_qa_chain(mode="text", top_k=top_k)
        result = qa({"query": query})
        logger.info("Text QA completed successfully.")
        return JSONResponse({
            "answer": result["result"],
            "evidence": [
                {"text": doc.page_content, "meta": doc.metadata}
                for doc in result["source_documents"]
            ]
        })
    except Exception as e:
        logger.error(f"Text QA failed: {e}")
        return JSONResponse({"error": "Text QA failed. See server logs for details."}, status_code=500)

@app.post("/query_audio")
async def query_audio(file: UploadFile = File(...), top_k: int = Form(5)):
    """
    Answer a question using an uploaded audio file, with evidence.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        qa = get_qa_chain(mode="audio", top_k=top_k)
        result = qa({"query": tmp_path})
        logger.info("Audio QA completed successfully.")
        return JSONResponse({
            "answer": result["result"],
            "evidence": [
                {"text": doc.page_content, "meta": doc.metadata}
                for doc in result["source_documents"]
            ]
        })
    except Exception as e:
        logger.error(f"Audio QA failed: {e}")
        return JSONResponse({"error": "Audio QA failed. See server logs for details."}, status_code=500)
