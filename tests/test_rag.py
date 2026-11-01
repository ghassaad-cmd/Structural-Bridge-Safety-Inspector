import pytest

from bridge_safety_inspector.rag import (
    DEFAULT_BRIDGE_KB,
    NO_INFORMATION_ANSWER,
    BridgeKnowledgeBase,
    KnowledgeEntry,
)


@pytest.mark.parametrize(
    ("question", "expected_title"),
    [
        ("How do I inspect visible concrete cracks?", "Concrete cracks"),
        ("How should crack monitoring records be reviewed?", "monitoring records"),
        ("What repair approaches are used in bridge maintenance?", "Maintenance"),
        ("What procedures are involved in a bridge field inspection?", "inspection"),
    ],
)
def test_search_retrieves_relevant_reference(question, expected_title):
    results = DEFAULT_BRIDGE_KB.search(question)

    assert results
    assert any(expected_title.lower() in entry.title.lower() for entry in results)


def test_answer_includes_retrieved_source_references():
    result = DEFAULT_BRIDGE_KB.answer("How should bridge inspections be documented?")

    assert result.sources
    assert "FHWA" in result.answer
    for source in result.sources:
        assert source.title
        assert source.url.startswith("https://")
        assert f"[Source: {source.title}]({source.url})" in result.answer


def test_unrelated_question_returns_clear_no_information_answer():
    result = DEFAULT_BRIDGE_KB.answer("What is the capital of France?")

    assert result.answer == NO_INFORMATION_ANSWER
    assert not result.sources


def test_query_with_only_a_broad_bridge_term_has_no_relevant_answer():
    result = DEFAULT_BRIDGE_KB.answer("What is the bridge's current load capacity?")

    assert result.answer == NO_INFORMATION_ANSWER
    assert not result.sources


def test_empty_or_stop_word_only_query_returns_no_results():
    assert DEFAULT_BRIDGE_KB.search("") == []
    assert DEFAULT_BRIDGE_KB.search("what is it") == []


def test_custom_entries_are_retrievable_and_answers_are_grounded():
    knowledge_base = BridgeKnowledgeBase(
        [
            KnowledgeEntry(
                title="Local guidance",
                content="This source says to document the observed crack location.",
                tags=("crack", "location"),
                source_name="Sample reference",
                source_url="https://example.org/reference",
            )
        ]
    )

    result = knowledge_base.answer("Where should a crack be documented?")

    assert "document the observed crack location" in result.answer
    assert result.sources[0].title == "Sample reference"


def test_answer_does_not_present_unsourced_entries_as_guidance():
    knowledge_base = BridgeKnowledgeBase(
        [
            KnowledgeEntry(
                title="Unverified note",
                content="Crack repair is needed.",
                tags=("crack", "repair"),
            )
        ]
    )

    result = knowledge_base.answer("What is the crack repair approach?")

    assert result.answer == NO_INFORMATION_ANSWER
    assert not result.sources


def test_search_with_nonpositive_top_k_returns_no_results():
    assert DEFAULT_BRIDGE_KB.search("bridge inspection", top_k=0) == []
