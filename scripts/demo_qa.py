import argparse
import logging
from voxarch.rag.langchain_retriever import VoxarchRetriever
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from voxarch.utils.config import Config

# Configure root logger for demo runs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("voxarch.scripts.demo_qa")

def load_llm(model_name):
    """
    Loads a HuggingFace language model for use in RetrievalQA.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        logger.info(f"Loaded HuggingFace LLM: {model_name}")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Failed to load HuggingFace LLM '{model_name}': {e}")
        raise

def main():
    """
    Command-line demo: runs LangChain RetrievalQA on a text or audio query and prints the answer and evidence.
    Logs all steps and errors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="voxarch/config/config.yaml")
    parser.add_argument('--mode', choices=["text", "audio"], default="text", help="Choose between text or audio query.")
    parser.add_argument('--query', required=True, help="Text query or path to audio file.")
    parser.add_argument('--top_k', type=int, default=None, help="Number of results to retrieve.")
    parser.add_argument('--hf_model', default=None, help="HuggingFace model repo or path (optional).")
    args = parser.parse_args()

    try:
        config = Config(args.config)
        model_name = args.hf_model or config.get("models.llm", "mistralai/Mistral-7B-Instruct-v0.2")
        top_k = args.top_k or config.get("search.top_k", 5)
        llm = load_llm(model_name)
        retriever = VoxarchRetriever(config_path=args.config, mode=args.mode, top_k=top_k)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info(f"Running RetrievalQA (mode={args.mode}, top_k={top_k})")
        result = qa({"query": args.query})
        print("\nANSWER:\n", result["result"])
        print("\nEVIDENCE:")
        for doc in result["source_documents"]:
            print("----")
            print("Text:", doc.page_content[:200].replace("\n", " "))
            print("Meta:", doc.metadata)
        logger.info("QA run completed successfully.")
    except Exception as e:
        logger.error(f"QA demo failed: {e}")

if __name__ == "__main__":
    main()
