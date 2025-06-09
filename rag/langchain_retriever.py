import logging
from langchain.schema import BaseRetriever, Document
from voxarch.rag.vectorstore import VectorStore

logger = logging.getLogger("voxarch.langchain_retriever")
logger.setLevel(logging.INFO)

class VoxarchRetriever(BaseRetriever):
    """
    LangChain retriever wrapping the Voxarch VectorStore.
    Supports both text and audio queries with provenance.
    Logs queries and errors.
    """
    def __init__(self, config_path="voxarch/config/config.yaml", mode="text", top_k=None):
        """
        Initializes the retriever.
        mode: "text" or "audio"
        top_k: number of results to retrieve
        """
        try:
            self.vs = VectorStore(config_path)
            self.vs.load()
            self.mode = mode
            self.top_k = top_k
            logger.info(f"Initialized VoxarchRetriever (mode={mode}, top_k={top_k})")
        except Exception as e:
            logger.error(f"Failed to initialize VoxarchRetriever: {e}")
            raise

    def _get_relevant_documents(self, query, **kwargs):
        """
        Retrieve top-k relevant Documents for a query.
        If mode is 'audio', query is a file path; otherwise, it is text.
        Logs errors and successful queries.
        """
        try:
            top_k = self.top_k or 5
            if self.mode == "audio":
                results = self.vs.query_audio(query, top_k=top_k)
            else:
                results = self.vs.query(query, top_k=top_k)
            docs = []
            for meta in results:
                docs.append(Document(
                    page_content=meta.get("text") or "",
                    metadata=meta
                ))
            logger.info(f"Retrieved {len(docs)} documents for query (mode={self.mode}).")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{str(query)[:40]}...': {e}")
            return []

    async def aget_relevant_documents(self, query, **kwargs):
        return self._get_relevant_documents(query, **kwargs)
