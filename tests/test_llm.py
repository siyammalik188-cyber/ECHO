"""Tests for the real AnthropicLLM class, with a stubbed SDK client.

These exercise ECHO's own request-building and response-parsing code. They do
not exercise the Anthropic API itself — see README, "What is not covered".
"""

from types import SimpleNamespace

import pytest

from echo.llm import FALLBACK_BETA, AnthropicLLM, LLMError


def block(kind, text=""):
    return SimpleNamespace(type=kind, text=text)


class StubSDK:
    """Mimics the shape of `anthropic.Anthropic` that AnthropicLLM touches."""

    def __init__(self, response):
        self.response = response
        self.request = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.request = kwargs
        return self.response


def make_llm(response):
    sdk = StubSDK(response)
    return AnthropicLLM(client=sdk), sdk


def test_text_blocks_are_joined_and_stripped():
    llm, _ = make_llm(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[block("text", "  hello "), block("text", "world  ")],
        )
    )
    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello world"


def test_non_text_blocks_are_ignored():
    llm, _ = make_llm(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[block("thinking"), block("text", "the answer")],
        )
    )
    assert llm.complete([{"role": "user", "content": "hi"}]) == "the answer"


def test_request_carries_model_messages_and_fallback_optin():
    llm, sdk = make_llm(
        SimpleNamespace(stop_reason="end_turn", content=[block("text", "ok")])
    )
    messages = [{"role": "user", "content": "hi"}]

    llm.complete(messages, system="be terse")

    assert sdk.request["model"] == llm.model
    assert sdk.request["messages"] == messages
    assert sdk.request["system"] == "be terse"
    assert sdk.request["betas"] == [FALLBACK_BETA]
    assert sdk.request["fallbacks"] == "default"


def test_system_is_omitted_when_absent():
    llm, sdk = make_llm(
        SimpleNamespace(stop_reason="end_turn", content=[block("text", "ok")])
    )
    llm.complete([{"role": "user", "content": "hi"}])
    assert "system" not in sdk.request


def test_a_refusal_raises_rather_than_returning_empty_text():
    llm, _ = make_llm(
        SimpleNamespace(
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber"),
            content=[],
        )
    )
    with pytest.raises(LLMError, match="cyber"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_an_empty_response_raises():
    llm, _ = make_llm(SimpleNamespace(stop_reason="max_tokens", content=[]))
    with pytest.raises(LLMError):
        llm.complete([{"role": "user", "content": "hi"}])


# ------------------------------------------------- structured output (extraction)


def test_structured_response_is_parsed_into_a_dict():
    llm, sdk = make_llm(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[block("text", '{"memories": [{"content": "x"}]}')],
        )
    )
    schema = {"type": "object", "properties": {}}

    payload = llm.complete_structured([{"role": "user", "content": "go"}], schema)

    assert payload == {"memories": [{"content": "x"}]}
    assert sdk.request["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }


def test_structured_requests_keep_the_fallback_optin():
    llm, sdk = make_llm(
        SimpleNamespace(stop_reason="end_turn", content=[block("text", "{}")])
    )
    llm.complete_structured([{"role": "user", "content": "go"}], {"type": "object"})

    assert sdk.request["betas"] == [FALLBACK_BETA]
    assert sdk.request["fallbacks"] == "default"


def test_non_json_in_a_structured_response_raises():
    llm, _ = make_llm(
        SimpleNamespace(stop_reason="end_turn", content=[block("text", "not json")])
    )
    with pytest.raises(LLMError, match="valid JSON"):
        llm.complete_structured([{"role": "user", "content": "go"}], {"type": "object"})


def test_a_json_array_where_an_object_was_expected_raises():
    llm, _ = make_llm(
        SimpleNamespace(stop_reason="end_turn", content=[block("text", "[1, 2]")])
    )
    with pytest.raises(LLMError, match="expected an object"):
        llm.complete_structured([{"role": "user", "content": "go"}], {"type": "object"})


def test_a_refusal_on_a_structured_request_still_raises():
    llm, _ = make_llm(
        SimpleNamespace(
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber"),
            content=[],
        )
    )
    with pytest.raises(LLMError, match="cyber"):
        llm.complete_structured([{"role": "user", "content": "go"}], {"type": "object"})
