# from dataclasses import dataclass
# from typing import List
# from langchain.prompts import ChatPromptTemplate
# from langchain_aws import ChatBedrock
# from rag_app.get_chroma_db import get_chroma_db

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


# @dataclass
# class QueryResponse:
#     query_text: str
#     response_text: str
#     sources: List[str]


# def query_rag(query_text: str) -> QueryResponse:
#     db = get_chroma_db()

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=3)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
#     response = model.invoke(prompt)
#     response_text = response.content

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     print(f"Response: {response_text}\nSources: {sources}")

#     return QueryResponse(
#         query_text=query_text, response_text=response_text, sources=sources
#     )


# if __name__ == "__main__":
#     query_rag("How much does a landing page cost to develop?")

from dataclasses import dataclass
from typing import List
from rag_app.interfaces import LLMInterface
from rag_app.interfaces import VectorStoreInterface
from rag_app.get_chroma_db import get_chroma_db  # Returns a VectorStoreInterface
from rag_app.bedrock_llm import BedrockLLM        # Implements LLMInterface
import sys

BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: List[str]

def query_rag(
    query_text: str,
    vectorstore: VectorStoreInterface,
    llm: LLMInterface,
    k: int = 3
) -> QueryResponse:
    # Vector search
    results = vectorstore.similarity_search_with_score(query_text, k=k)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Generate response from LLM
    response_text = llm.invoke(context=context_text, question=query_text)

    # Collect metadata
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"Response: {response_text}\nSources: {sources}")

    return QueryResponse(
        query_text=query_text,
        response_text=response_text,
        sources=sources
    )

if __name__ == "__main__":
    # These are the interface implementations that should be defined in your application.
    chroma_db = get_chroma_db()  # Must return VectorStoreInterface
    llm = BedrockLLM(model_id=BEDROCK_MODEL_ID)
    prompt = sys.argv[1]
    query_rag(prompt, chroma_db, llm)
