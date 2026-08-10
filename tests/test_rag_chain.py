import pytest
from app.services.rag_chain import RAGChain


def test_rag_chain_refusal_when_insufficient_evidence():
    chain = RAGChain()
    result = chain.generate_response("What is quantum entanglement in cardiology?", contexts=[])

    assert result["grounded"] is False
    assert result["sources"] == []
    assert "could not find sufficient evidence" in result["answer"].lower()


def test_rag_chain_offline_fallback():
    # Pass an invalid Ollama port/URL to trigger connection refusal
    chain = RAGChain(ollama_url="http://localhost:99999")
    contexts = [
        {
            "parent_id": "p1",
            "text": "Carbamazepine is indicated for trigeminal neuralgia.",
            "metadata": {"source": "pharma.pdf", "page": 42, "heading": "Neuralgia Management"},
        }
    ]
    result = chain.generate_response("What treats trigeminal neuralgia?", contexts=contexts)

    assert result["grounded"] is False
    assert result["sources"] == []
    assert "currently unavailable" in result["answer"].lower()
    assert "warning" in result


def test_rag_prompt_construction():
    chain = RAGChain()
    contexts = [
        {
            "parent_id": "p1",
            "text": "Carbamazepine is indicated for trigeminal neuralgia.",
            "metadata": {"source": "pharma.pdf", "page": 42, "heading": "Neuralgia Management"},
        }
    ]
    prompt = chain.build_user_prompt("What treats trigeminal neuralgia?", contexts)
    assert "pharma.pdf" in prompt
    assert "Page 42" in prompt
    assert "Neuralgia Management" in prompt
