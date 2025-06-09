import logging
from voxarch.rag.vectorstore import VectorStore

# Set up root logger (prints to console)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

if __name__ == "__main__":
    try:
        store = VectorStore()
        store.build()
        store.save()
        logging.info("Index built and saved successfully.")
    except Exception as e:
        logging.error(f"Index build failed: {e}")
