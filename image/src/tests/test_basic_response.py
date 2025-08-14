from rag_app.get_chroma_db import get_chroma_db  # Returns a VectorStoreInterface
from rag_app.bedrock_llm import BedrockLLM        # Implements LLMInterface
from rag_app.query_rag import query_rag
from dotenv import load_dotenv
from pathlib import Path

import os
import json
import pytest

from deepeval.test_case import LLMTestCase
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.toxicity.toxicity import ToxicityMetric
from deepeval.metrics.summarization.summarization import SummarizationMetric
from deepeval.metrics.hallucination.hallucination import HallucinationMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric

BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

# Load prompts.json
# Path to prompts.json sitting next to this test file
prompts_path = Path(__file__).resolve().with_name("prompts.json")
if not prompts_path.exists():
    raise FileNotFoundError(f"prompts.json not found at: {prompts_path}")

with prompts_path.open("r", encoding="utf-8") as f:
    test_scenarios = json.load(f)

# Initialize RAG pipeline components
chroma_db = get_chroma_db()  # Must return VectorStoreInterface
llm = BedrockLLM(model_id=BEDROCK_MODEL_ID)

@pytest.mark.parametrize("scenario", test_scenarios)
def test_llm_answer_relevance(scenario):
    prompt = scenario["prompt"]
    expected_output = scenario["expected_output"]

    actual_output = query_rag(prompt, chroma_db, llm).response_text

    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_output
    )

    metric = AnswerRelevancyMetric()
    metric.measure(test_case)
    assert getattr(test_case, "passed", True), f"Relevancy flagged for: {prompt}\nOutput: {actual}"

@pytest.mark.parametrize("scenario", test_scenarios)
def test_toxicity(scenario):
    prompt = scenario["prompt"]
    expected = scenario["expected_output"]
    actual = query_rag(prompt, chroma_db, llm).response_text

    test_case = LLMTestCase(input=prompt, actual_output=actual, expected_output=expected)
    metric = ToxicityMetric()

    if hasattr(metric, "use_metric"):
        metric.use_metric(test_case)
    elif hasattr(metric, "measure"):
        metric.measure(test_case)
    else:
        pytest.skip("Metric object missing use_metric/measure on this version.")

    # Expect low toxicity—fail if flagged
    assert getattr(test_case, "passed", True), f"Toxicity flagged for: {prompt}\nOutput: {actual}"

@pytest.mark.parametrize("scenario", test_scenarios)
def test_summarization(scenario):
    prompt = scenario["prompt"]
    expected = scenario["expected_output"]
    actual = query_rag(prompt, chroma_db, llm).response_text

    test_case = LLMTestCase(input=prompt, actual_output=actual, expected_output=expected)
    metric = SummarizationMetric()

    if hasattr(metric, "use_metric"):
        metric.use_metric(test_case)
    elif hasattr(metric, "measure"):
        metric.measure(test_case)
    else:
        pytest.skip("Metric object missing use_metric/measure on this version.")

    # Expect low toxicity—fail if flagged
    assert getattr(test_case, "passed", True), f"Summarization flagged for: {prompt}\nOutput: {actual}"

@pytest.mark.parametrize("scenario", test_scenarios)
def test_hallucination(scenario):
    prompt = scenario["prompt"]
    expected = scenario["expected_output"]

    # Run your pipeline
    rag_result = query_rag(prompt, chroma_db, llm)
    actual = rag_result.response_text

    # 1) Try to get contexts from your pipeline result (if your code exposes them)
    contexts = getattr(rag_result, "contexts", None)

    # 2) Fallback: fetch top-k docs from your vector store
    if not contexts:
        try:
            # Chroma/LangChain style: returns list[Document] with .page_content
            docs = chroma_db.similarity_search(prompt, k=5)
            contexts = [d.page_content for d in docs]
        except AttributeError:
            # If your VectorStoreInterface exposes a different API, adapt here:
            # e.g., contexts = chroma_db.retrieve_texts(prompt, top_k=5)
            pytest.skip("Could not retrieve contexts; ensure your vector store exposes a retrieval method.")

    # Make sure contexts is a non-empty list[str]
    if not contexts or not isinstance(contexts, list):
        pytest.skip("No contexts available; HallucinationMetric requires non-empty context.")

    # Optional: keep contexts short to avoid judge model token limits
    contexts = [c if len(c) < 4000 else c[:4000] for c in contexts]

    case = RAGTestCase(
        query=prompt,
        context=contexts,              # <- REQUIRED for hallucination/faithfulness metrics
        actual_output=actual,
        expected_output=expected
    )

    metric = HallucinationMetric()
    # v3+ API
    if hasattr(metric, "use_metric"):
        metric.use_metric(case)
    elif hasattr(metric, "measure"):
        metric.measure(case)
    else:
        pytest.skip("Metric object missing use_metric/measure on this version.")

    assert getattr(case, "passed", True), (
        f"Hallucination detected for prompt: {prompt}\n"
        f"Output: {actual}\n"
        f"Contexts (top {len(contexts)}): {contexts[:2]}{' ...' if len(contexts) > 2 else ''}"
    )