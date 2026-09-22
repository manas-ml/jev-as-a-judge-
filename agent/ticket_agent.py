"""Support-ticket resolution agent: a Strands Agent with one deterministic KB-search tool.

This module is intentionally free of judge/eval logic - it only knows how to
answer a ticket and how to report what it did (the KB queries it ran and the
KB entries it saw), so eval/run_agent.py can freeze that output for judging.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from strands import Agent, tool
from strands.models.openai import OpenAIModel

from eval.common import require_env

KB_PATH = Path(__file__).resolve().parent / "kb.json"

with open(KB_PATH) as f:
    _KB: list[dict[str, str]] = json.load(f)

_STOPWORDS = TfidfVectorizer(stop_words="english").get_stop_words()
_STEMMER = PorterStemmer()


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip to word characters, drop stopwords, and Porter-stem.

    Stemming matters here: without it "charge"/"charged"/"charges" are three
    unrelated tokens under plain TF-IDF, which tanks recall on a small fixed KB.
    Deterministic (no randomness, no external calls) so repeated searches for the
    same query always return the same result.
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [_STEMMER.stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1]


_KB_DOCS = [f"{entry['title']}. {entry['text']}" for entry in _KB]
_VECTORIZER = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None)
_KB_MATRIX = _VECTORIZER.fit_transform(_KB_DOCS)

SYSTEM_PROMPT = """You are a support agent for a SaaS product's billing and account help desk.

Rules:
- For any question about billing, payments, subscriptions, accounts, or similar account-management
  topics, you MUST call the kb_search tool first and base your answer only on the KB entries it
  returns before answering.
- If the KB results do not actually cover the customer's issue, say so plainly and tell the
  customer you are escalating this to a human agent. Do not guess, speculate, or invent policy
  that is not in the KB.
- Keep answers concise and directly actionable.
- If the question is a simple greeting or clearly has nothing to do with billing/account topics,
  you may answer directly without searching the KB.
"""


@tool
def kb_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search the fixed billing/account FAQ knowledge base for entries relevant to a query.

    Deterministic TF-IDF cosine-similarity search over a fixed KB - the same query always
    returns the same results, so repeated agent runs on the same ticket stay reproducible.

    Args:
        query: The customer's question or a short paraphrase of the issue.
        top_k: Maximum number of KB entries to return.

    Returns:
        A list of matched KB entries (id, title, text) each with a similarity score, sorted
        highest score first.
    """
    query_vec = _VECTORIZER.transform([query])
    scores = cosine_similarity(query_vec, _KB_MATRIX)[0]
    ranked = sorted(range(len(_KB)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            continue
        results.append({**_KB[i], "score": round(float(scores[i]), 4)})
    return results


def build_agent() -> Agent:
    model = OpenAIModel(
        client_args={
            "api_key": require_env("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
        },
        model_id=require_env("AGENT_MODEL_ID"),
    )
    return Agent(model=model, tools=[kb_search], system_prompt=SYSTEM_PROMPT)


def _text_of(block: dict[str, Any]) -> str | None:
    return block.get("text")


def run_ticket(question: str) -> dict[str, Any]:
    """Run a single ticket through a fresh agent instance and extract a frozen record.

    A fresh Agent is created per ticket so `agent.messages` only ever contains
    the conversation for this one question - no cross-ticket history to filter out.

    Returns a dict with: answer, tool_calls (query sent to kb_search per call),
    kb_evidence (flattened KB entries returned across all tool calls).
    """
    agent = build_agent()
    result = agent(question)
    answer = str(result).strip()

    tool_calls: list[dict[str, Any]] = []
    kb_evidence: list[dict[str, Any]] = []
    tool_use_by_id: dict[str, dict[str, Any]] = {}

    for message in agent.messages:
        for block in message.get("content", []):
            tool_use = block.get("toolUse")
            if tool_use and tool_use.get("name") == "kb_search":
                tool_use_by_id[tool_use["toolUseId"]] = tool_use
                tool_calls.append({"query": tool_use.get("input", {}).get("query"),
                                    "top_k": tool_use.get("input", {}).get("top_k")})

            tool_result = block.get("toolResult")
            if tool_result and tool_result.get("toolUseId") in tool_use_by_id:
                for content_item in tool_result.get("content", []):
                    payload = content_item.get("json")
                    if payload is None and "text" in content_item:
                        try:
                            payload = json.loads(content_item["text"])
                        except (json.JSONDecodeError, TypeError):
                            payload = None
                    if isinstance(payload, list):
                        kb_evidence.extend(payload)

    # De-duplicate KB evidence by id (a repeated identical search across turns is possible).
    seen_ids = set()
    deduped_evidence = []
    for entry in kb_evidence:
        entry_id = entry.get("id")
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        deduped_evidence.append(entry)

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "kb_evidence": deduped_evidence,
    }
