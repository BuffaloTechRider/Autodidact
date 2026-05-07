"""Tests for LearningExtractor — structured knowledge extraction from cloud responses.

TDD: tests written first, then implementation.

The LearningExtractor takes a question + cloud answer and extracts structured
knowledge entries (and optionally skills) via a local LLM call. Falls back to
storing the raw Q&A pair if extraction fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autodidact.llm_client import ChatResponse, LLMClient


# ── Test: successful extraction ──────────────────────────────────

class TestSuccessfulExtraction:
    """When the LLM returns valid JSON, extract structured knowledge entries."""

    def test_extracts_knowledge_entries_from_valid_json(self):
        """LLM returns clean JSON → multiple knowledge entries extracted."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content='{"knowledge": [{"content": "Python was created in 1991", "confidence": 0.95}, {"content": "Guido van Rossum created Python", "confidence": 0.9}], "skills": []}',
            model="qwen2.5:7b",
            input_tokens=200,
            output_tokens=50,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract(
            question="Who created Python and when?",
            answer="Python was created by Guido van Rossum in 1991.",
        )

        assert len(result.knowledge) == 2
        assert result.knowledge[0].content == "Python was created in 1991"
        assert result.knowledge[0].confidence == 0.95
        assert result.knowledge[0].source == "cloud_escalation"
        assert result.knowledge[1].content == "Guido van Rossum created Python"

    def test_extracts_skills_when_present(self):
        """LLM returns skills with steps → skill entries extracted."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content='{"knowledge": [{"content": "Docker containers are lightweight", "confidence": 0.8}], "skills": [{"name": "deploy_docker", "description": "Deploy with Docker", "steps": [{"order": 1, "description": "Write Dockerfile"}, {"order": 2, "description": "Build image"}, {"order": 3, "description": "Run container"}]}]}',
            model="qwen2.5:7b",
            input_tokens=300,
            output_tokens=100,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract(
            question="How do I deploy with Docker?",
            answer="First write a Dockerfile, then build the image, then run the container.",
        )

        assert len(result.knowledge) == 1
        assert len(result.skills) == 1
        assert result.skills[0]["name"] == "deploy_docker"
        assert len(result.skills[0]["steps"]) == 3


# ── Test: JSON parsing resilience ────────────────────────────────

class TestJsonParsing:
    """The extractor should handle various LLM output formats."""

    def test_handles_markdown_code_block_wrapper(self):
        """LLM wraps JSON in ```json ... ``` → still parses."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content='```json\n{"knowledge": [{"content": "fact one", "confidence": 0.8}], "skills": []}\n```',
            model="qwen2.5:7b",
            input_tokens=100,
            output_tokens=50,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract("q", "a")
        assert len(result.knowledge) == 1
        assert result.knowledge[0].content == "fact one"

    def test_handles_json_embedded_in_text(self):
        """LLM returns text before/after JSON → extracts the JSON object."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content='Here is the extraction:\n{"knowledge": [{"content": "extracted fact", "confidence": 0.7}], "skills": []}\nDone.',
            model="qwen2.5:7b",
            input_tokens=100,
            output_tokens=50,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract("q", "a")
        assert len(result.knowledge) == 1
        assert result.knowledge[0].content == "extracted fact"


# ── Test: fallback behavior ──────────────────────────────────────

class TestFallback:
    """When extraction fails, fall back to storing the raw Q&A pair."""

    def test_falls_back_on_invalid_json(self):
        """LLM returns garbage → fallback to raw answer as single entry."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content="I cannot extract structured data from this.",
            model="qwen2.5:7b",
            input_tokens=100,
            output_tokens=20,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract(
            question="What is quantum computing?",
            answer="Quantum computing uses qubits to perform calculations.",
        )

        # Should fall back to a single knowledge entry with the raw answer.
        assert len(result.knowledge) == 1
        assert "qubits" in result.knowledge[0].content
        assert result.knowledge[0].source == "cloud_escalation"
        assert result.knowledge[0].confidence <= 0.7  # lower confidence for fallback

    def test_falls_back_on_llm_exception(self):
        """LLM call throws → fallback to raw answer."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.side_effect = Exception("Connection refused")

        extractor = LearningExtractor(mock_client)
        result = extractor.extract(
            question="What is quantum computing?",
            answer="Quantum computing uses qubits.",
        )

        assert len(result.knowledge) == 1
        assert "qubits" in result.knowledge[0].content

    def test_falls_back_on_empty_knowledge_array(self):
        """LLM returns valid JSON but empty knowledge → fallback to raw."""
        from autodidact.learning_extractor import LearningExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content='{"knowledge": [], "skills": []}',
            model="qwen2.5:7b",
            input_tokens=100,
            output_tokens=20,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract("q", "Some answer text here")

        assert len(result.knowledge) == 1
        assert "Some answer text here" in result.knowledge[0].content


# ── Test: content limits ─────────────────────────────────────────

class TestContentLimits:
    """Extracted content should be bounded to prevent bloat."""

    def test_truncates_long_content(self):
        """Knowledge content longer than 500 chars gets truncated."""
        from autodidact.learning_extractor import LearningExtractor

        long_content = "x" * 1000
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat.return_value = ChatResponse(
            content=f'{{"knowledge": [{{"content": "{long_content}", "confidence": 0.8}}], "skills": []}}',
            model="qwen2.5:7b",
            input_tokens=100,
            output_tokens=50,
        )

        extractor = LearningExtractor(mock_client)
        result = extractor.extract("q", "a")
        assert len(result.knowledge[0].content) <= 500


# ── Test: ExtractionResult structure ─────────────────────────────

class TestExtractionResult:
    """The ExtractionResult dataclass has the right shape."""

    def test_extraction_result_has_knowledge_and_skills(self):
        from autodidact.learning_extractor import ExtractionResult
        from autodidact.types import NewKnowledgeEntry

        result = ExtractionResult(
            knowledge=[NewKnowledgeEntry(content="fact", source="test")],
            skills=[],
        )
        assert len(result.knowledge) == 1
        assert len(result.skills) == 0
