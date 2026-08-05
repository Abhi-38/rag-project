import requests
import json
from typing import List, Dict, Any
from app.core.config import settings

class RAGChain:
    """Manages medical context synthesis, LLM generation, and citation formatting."""

    SYSTEM_PROMPT = """You are a specialized Medical AI Assistant providing accurate, grounded answers based strictly on the provided medical study documents.

STRICT RULES:
1. Answer ONLY using the facts explicitly stated in the provided Context Snippets.
2. Do NOT assume, extrapolate, or bring in outside medical knowledge.
3. If the answer cannot be directly determined from the provided context, state clearly: "I cannot find sufficient evidence in the provided medical documents to answer this question."
4. Every clinical statement or recommendation MUST include source citations referencing the Page and Chapter/Heading provided in the context tags.

Format your response clearly using bullet points and section headings where appropriate.
"""

    def __init__(self, ollama_url: str = settings.OLLAMA_BASE_URL, model: str = settings.LLM_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def build_user_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Formats query and context snippets into a grounded LLM prompt."""
        context_str = ""
        for idx, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            page = meta.get("page", "Unknown")
            source = meta.get("source", "Medical Document")
            heading = meta.get("heading", "General")

            context_str += f"\n--- Context Snippet {idx} ---\n"
            context_str += f"Source: {source} | Page: {page} | Section: {heading}\n"
            context_str += f"Content:\n{ctx['text']}\n"

        prompt = f"Medical Query: {query}\n\nContext Snippets:\n{context_str}\n\nAnswer:"
        return prompt

    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates grounded response using Ollama local API or structured fallback."""
        if not contexts:
            return {
                "answer": "No relevant medical context was found in the indexed documents.",
                "sources": [],
                "grounded": False
            }

        prompt = self.build_user_prompt(query, contexts)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": settings.LLM_TEMPERATURE
            },
            "stream": False
        }

        sources = [
            {
                "source": c.get("metadata", {}).get("source"),
                "page": c.get("metadata", {}).get("page"),
                "heading": c.get("metadata", {}).get("heading")
            }
            for c in contexts
        ]

        try:
            res = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            if res.status_code == 200:
                answer = res.json().get("message", {}).get("content", "")
                return {
                    "answer": answer,
                    "sources": sources,
                    "grounded": True
                }
        except Exception as e:
            # If local Ollama endpoint is offline/not running, return structured context synthesis fallback
            fallback_answer = f"**[Ollama LLM Offline]** - Synthesized context retrieved from medical sources:\n\n"
            for idx, c in enumerate(contexts, 1):
                meta = c.get("metadata", {})
                fallback_answer += f"**Snippet {idx} (Page {meta.get('page')}, {meta.get('heading')}):**\n{c['text'][:300]}...\n\n"

            return {
                "answer": fallback_answer,
                "sources": sources,
                "grounded": True,
                "warning": f"LLM Connection Error: {str(e)}"
            }

        return {
            "answer": "Failed to generate LLM response.",
            "sources": sources,
            "grounded": False
        }
