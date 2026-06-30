import os

import pytest

from llm import LLAMA_AVAILABLE, extract_knowledge
from tests.evals.fixtures import EVAL_CASES

MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")
EVAL_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", "0.7"))

pytestmark = pytest.mark.llm


def _jaccard(expected, actual):
    expected_set, actual_set = set(expected), set(actual)
    if not expected_set and not actual_set:
        return 1.0
    union = expected_set | actual_set
    if not union:
        return 1.0
    return len(expected_set & actual_set) / len(union)


@pytest.fixture(scope="module")
def llm_instance():
    if not LLAMA_AVAILABLE:
        pytest.skip("llama-cpp-python not installed")
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model not found at {MODEL_PATH}; set EVAL_MODEL_PATH")
    from llama_cpp import Llama
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, use_mlock=False, verbose=False)


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_extraction_case(case, llm_instance):
    if "xfail" in case:
        pytest.xfail(case["xfail"])

    actual = extract_knowledge(case["game_output"], case["action"], llm_instance)

    for field, forbidden in case.get("must_not", {}).items():
        actual_values = actual.get(field, [])
        violations = set(forbidden) & set(actual_values)
        assert not violations, f"{case['id']}: forbidden values {violations} found in '{field}': {actual_values}"

    for field in case.get("absent", []):
        value = actual.get(field)
        assert not value, f"{case['id']}: field '{field}' should be absent/null, got: {value!r}"

    scores = []
    for field, expected_value in case["expected"].items():
        actual_value = actual.get(field, [])
        scores.append(_jaccard(expected_value, actual_value))
    score = sum(scores) / len(scores) if scores else 1.0

    assert score >= EVAL_THRESHOLD, f"{case['id']}: score {score:.2f} below threshold {EVAL_THRESHOLD}. Got: {actual}"
