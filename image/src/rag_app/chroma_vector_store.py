from rag_app.interfaces import VectorStoreInterface
from langchain_community.vectorstores import Chroma
from typing import List, Tuple
from langchain.schema.document import Document

class ChromaVectorStore(VectorStoreInterface):
    def __init__(self, chroma_instance: Chroma):
        self.chroma = chroma_instance

    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.chroma.similarity_search_with_score(query, k=k)
