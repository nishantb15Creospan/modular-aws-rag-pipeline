from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain.schema.document import Document

class VectorStoreInterface(ABC):
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        pass

class EmbeddingFunctionInterface(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

class LLMInterface(ABC):
    @abstractmethod
    def invoke(self, context: str, question: str) -> str:
        pass
