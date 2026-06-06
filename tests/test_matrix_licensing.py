from pour_decisions.matrix.licensing import enforce_derivative_name, notice_for
from pour_decisions.matrix.registry import by_key


def test_apache_has_no_notice():
    assert notice_for(by_key("qwen2.5-0.5b")) is None


def test_gemma_notice_mentions_terms_url():
    note = notice_for(by_key("gemma-3-270m"))
    assert note is not None and "ai.google.dev/gemma/terms" in note


def test_llama_notice_mentions_community_license():
    note = notice_for(by_key("llama-3.2-1b"))
    assert note is not None and "Llama 3.2 Community License" in note


def test_llama_derivative_name_is_prefixed():
    assert enforce_derivative_name(by_key("llama-3.2-1b"), "pour-decisions").startswith("Llama")


def test_non_llama_name_unchanged():
    assert enforce_derivative_name(by_key("qwen2.5-0.5b"), "pour-decisions") == "pour-decisions"
