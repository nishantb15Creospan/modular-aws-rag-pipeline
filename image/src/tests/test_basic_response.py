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

@pytest.mark.parametrize("scenario", test_scenarios)
def test_toxicity(scenario):
    prompt = scenario["prompt"]
    expected = scenario["expected_output"]
    actual = query_rag(prompt, chroma_db, llm).response_text

    case = LLMTestCase(input=prompt, actual_output=actual, expected_output=expected)
    metric = ToxicityMetric()

    if hasattr(metric, "use_metric"):
        metric.use_metric(case)
    elif hasattr(metric, "measure"):
        metric.measure(case)
    else:
        pytest.skip("Metric object missing use_metric/measure on this version.")

    # Expect low toxicity—fail if flagged
    assert getattr(case, "passed", True), f"Toxicity flagged for: {prompt}\nOutput: {actual}"