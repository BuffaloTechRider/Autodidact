"""LearningExtractor — structured knowledge extraction from cloud responses.

When the agent escalates to cloud, the extractor takes the question + answer
and extracts structured knowledge entries (facts) and optionally skills
(step-by-step procedures) via a local LLM call.

Falls back to storing the raw Q&A pair if extraction fails.

Ported from demo-prototype/src/learning-extractor.ts, simplified for Python.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from autodidact.llm_client import ChatMessage, LLMClient
from autodidact.types import NewKnowledgeEntry

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Extract knowledge and skills from the answer below. Return a JSON object with two arrays.

Rules:
- "knowledge": factual claims. Each: {"content": "one fact", "confidence": 0.9}
- "skills": step-by-step procedures found in the answer. Each: {"name": "short_snake_case_name", "description": "what this procedure does", "steps": [{"order": 1, "description": "step description"}]}
- Only include "skills" if the answer contains a clear multi-step procedure (3+ steps)
- Return ONLY valid JSON, nothing else

Example:
{"knowledge": [{"content": "Python was created in 1991", "confidence": 0.95}], "skills": []}

If no procedures found, return empty skills array: {"knowledge": [...], "skills": []}"""

_MAX_CONTENT_LENGTH = 500

_DOCUMENT_EXTRACTION_PROMPT = """Extract the key facts and concepts from this document section that would be valuable to remember without re-reading the source.

Rules:
- "knowledge": each entry is one fact/concept. {"content": "concise statement", "question": "a natural question this answers", "confidence": 0.9}
- Focus on information someone would want to RECALL later — not trivial details
- The "question" field is what a user might ask that this fact answers
- Skip boilerplate, table of contents, and purely structural content
- Return ONLY valid JSON, nothing else

Example:
{"knowledge": [{"content": "The API rate limit is 100 requests per minute per user", "question": "What is the API rate limit?", "confidence": 0.95}]}

If nothing worth extracting, return: {"knowledge": []}"""


@dataclass
class ExtractionResult:
    """Result of extracting knowledge and skills from a cloud response."""

    knowledge: list[NewKnowledgeEntry] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)


class LearningExtractor:
    """Extracts structured knowledge from cloud escalation responses."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def extract(self, question: str, answer: str) -> ExtractionResult:
        """Extract knowledge entries and skills from a cloud response.

        Parameters
        ----------
        question
            The user's original question.
        answer
            The cloud model's response.

        Returns
        -------
        ExtractionResult
            Structured knowledge and skills. Always contains at least one
            knowledge entry (falls back to raw answer on failure).
        """
        try:
            resp = self._client.chat(
                [
                    ChatMessage(role="system", content=_EXTRACTION_PROMPT),
                    ChatMessage(role="user", content=f"Question: {question}\n\nAnswer: {answer}"),
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            parsed = _parse_json(resp.content)
            if parsed is not None:
                result = _build_result(parsed, question)
                if result.knowledge:
                    return result
        except Exception as e:
            logger.warning("LearningExtractor: extraction failed: %s", e)

        # Fallback: store raw answer as a single entry.
        return _fallback_result(question, answer)

    def extract_from_document(self, section_text: str, source_file: str) -> ExtractionResult:
        """Extract knowledge from a document section during ingest.

        Unlike extract() which processes cloud Q&A pairs, this processes raw
        document text and produces knowledge entries with source='document_ingest'.
        """
        if not section_text.strip():
            return ExtractionResult()

        try:
            resp = self._client.chat(
                [
                    ChatMessage(role="system", content=_DOCUMENT_EXTRACTION_PROMPT),
                    ChatMessage(role="user", content=section_text[:4000]),
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            parsed = _parse_json(resp.content)
            if parsed is not None:
                result = _build_document_result(parsed, source_file)
                if result.knowledge:
                    return result
        except Exception as e:
            logger.warning("LearningExtractor: document extraction failed: %s", e)

        return ExtractionResult()


def _parse_json(content: str) -> dict[str, Any] | None:
    """Try multiple strategies to extract JSON from LLM output."""
    raw = content.strip()

    # Strategy 1: direct parse.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code block.
    stripped = re.sub(r"^```(?:json)?\s*", "", raw)
    stripped = re.sub(r"\s*```\s*$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first { ... } in the text.
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _build_result(parsed: dict[str, Any], question: str) -> ExtractionResult:
    """Build an ExtractionResult from parsed JSON."""
    knowledge: list[NewKnowledgeEntry] = []
    skills: list[dict] = []

    # Extract knowledge entries.
    raw_knowledge = parsed.get("knowledge", [])
    if isinstance(raw_knowledge, list):
        for item in raw_knowledge:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            content = str(item["content"])[:_MAX_CONTENT_LENGTH]
            confidence = item.get("confidence", 0.8)
            if not isinstance(confidence, (int, float)):
                confidence = 0.8
            confidence = max(0.0, min(1.0, float(confidence)))

            knowledge.append(NewKnowledgeEntry(
                content=content,
                source="cloud_escalation",
                confidence=confidence,
                domain="general",
                topic="learned",
                metadata={"extracted_from": question[:200]},
            ))

    # Extract skills.
    raw_skills = parsed.get("skills", [])
    if isinstance(raw_skills, list):
        for item in raw_skills:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            steps = item.get("steps", [])
            if not name or not isinstance(steps, list) or len(steps) < 2:
                continue
            skills.append({
                "name": name,
                "description": item.get("description", name),
                "steps": steps,
            })

    return ExtractionResult(knowledge=knowledge, skills=skills)


def _build_document_result(parsed: dict[str, Any], source_file: str) -> ExtractionResult:
    """Build an ExtractionResult from parsed document extraction JSON."""
    knowledge: list[NewKnowledgeEntry] = []

    raw_knowledge = parsed.get("knowledge", [])
    if isinstance(raw_knowledge, list):
        for item in raw_knowledge:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            content = str(item["content"])[:_MAX_CONTENT_LENGTH]
            confidence = item.get("confidence", 0.8)
            if not isinstance(confidence, (int, float)):
                confidence = 0.8
            confidence = max(0.0, min(1.0, float(confidence)))
            question = item.get("question")
            if isinstance(question, str):
                question = question[:200]
            else:
                question = None

            knowledge.append(NewKnowledgeEntry(
                content=content,
                question=question,
                # Uses "cloud_escalation" to satisfy the existing CHECK constraint.
                # Distinguished from real escalations by metadata.synthesized=True.
                # v1.5 will add "document_ingest" to the constraint.
                source="cloud_escalation",
                confidence=confidence,
                domain="general",
                topic="learned",
                metadata={"source_file": source_file, "synthesized": True},
            ))

    return ExtractionResult(knowledge=knowledge, skills=[])


def _fallback_result(question: str, answer: str) -> ExtractionResult:
    """Create a fallback result with the raw answer as a single entry."""
    return ExtractionResult(
        knowledge=[NewKnowledgeEntry(
            content=answer[:_MAX_CONTENT_LENGTH],
            source="cloud_escalation",
            confidence=0.7,
            domain="general",
            topic="learned",
            metadata={"extracted_from": question[:200], "fallback": True},
        )],
        skills=[],
    )
