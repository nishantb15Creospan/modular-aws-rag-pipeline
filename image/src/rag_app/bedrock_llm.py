from rag_app.interfaces import LLMInterface
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class BedrockLLM(LLMInterface):
    def __init__(self, model_id: str):
        self.model = ChatBedrock(model_id=model_id)

    def invoke(self, context: str, question: str) -> str:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=question)
        print(f"🧠 Prompt:\n{prompt}")
        response = self.model.invoke(prompt)
        return response.content
