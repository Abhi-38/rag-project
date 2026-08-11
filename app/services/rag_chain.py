from typing import Any, Dict, List
import requests

from app.core.config import settings


class RAGChain:
    """Manages medical context synthesis, LLM generation, citation formatting, and refusal handling."""

    SYSTEM_PROMPT = """You are a specialized Medical AI Assistant providing accurate, grounded answers based strictly on the provided medical study documents.

STRICT RULES:
1. Answer ONLY using the facts explicitly stated in the provided Context Snippets.
2. Do NOT assume, extrapolate, or bring in outside medical knowledge.
3. If the answer cannot be directly determined from the provided context, state clearly: "I could not find sufficient evidence in the uploaded medical documents to answer this question."
4. Every clinical statement or recommendation MUST include source citations referencing the Page and Chapter/Heading provided in the context tags.

Format your response clearly using bullet points and section headings where appropriate.
"""

    def __init__(self, ollama_url: str = settings.OLLAMA_BASE_URL, model: str = settings.LLM_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    @staticmethod
    def is_greeting(query: str) -> bool:
        """Determines if query is a conversational greeting or general assistant introduction."""
        cleaned = query.strip().lower().rstrip(".!?")
        greetings = {
            "hello", "hi", "hey", "greetings", "good morning",
            "good afternoon", "good evening", "howdy", "hola",
            "who are you", "what are you", "help", "who are u", "hi there", "hello there"
        }
        if cleaned in greetings or cleaned.startswith(("hi ", "hello ", "hey ", "greetings ")):
            return True
        return False

    def build_user_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Formats query and context snippets into a grounded LLM prompt."""
        context_str = ""
        for idx, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            page = meta.get("page")
            page_label = f"Page {page}" if page is not None else "Doc"
            source = meta.get("source", "Medical Document")
            heading = meta.get("heading", "General")

            context_str += f"\n--- Context Snippet {idx} ---\n"
            context_str += f"Source: {source} | {page_label} | Section: {heading}\n"
            context_str += f"Content:\n{ctx['text']}\n"

        prompt = f"Medical Query: {query}\n\nContext Snippets:\n{context_str}\n\nAnswer:"
        return prompt

    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates grounded response using Ollama local API or structured refusal/error handling."""
        # 1. Greeting & conversational intent handler
        if self.is_greeting(query):
            return {
                "answer": "Hello! I am your private Medical AI Assistant. I can help answer clinical questions, explain medical concepts, and summarize topics directly from your uploaded MBBS textbooks and study documents. How can I assist your studies today?",
                "sources": [],
                "grounded": True,
            }

        # 2. Evidence sufficiency check refusal
        if not contexts:
            return {
                "answer": "I could not find sufficient evidence in the uploaded medical documents to answer this question.",
                "sources": [],
                "grounded": False,
            }

        prompt = self.build_user_prompt(query, contexts)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
            },
            "stream": False,
        }

        sources = [
            {
                "source": c.get("metadata", {}).get("source"),
                "page": c.get("metadata", {}).get("page"),
                "heading": c.get("metadata", {}).get("heading"),
            }
            for c in contexts
        ]

        try:
            # Increase HTTP timeout to 300s (5 min) for local CPU LLM inference
            res = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=300)
            if res.status_code == 200:
                answer = res.json().get("message", {}).get("content", "")
                return {
                    "answer": answer,
                    "sources": sources,
                    "grounded": True,
                }
        except Exception as e:
            # Clear system error when LLM is unavailable; do NOT present raw context text as answer
            return {
                "answer": "The language model is currently processing or unavailable. Please try again later.",
                "sources": [],
                "grounded": False,
                "warning": f"LLM Connection Error: {str(e)}",
            }

        return {
            "answer": "The language model is currently processing or unavailable. Please try again later.",
            "sources": [],
            "grounded": False,
        }
