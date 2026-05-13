"""Tests for the CLI module.

Tests use mocked Agent/KnowledgeStore so they run without Ollama or cloud access.
Uses typer.testing.CliRunner for CLI invocation.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from autodidact.agent import QueryResponse, SavingsReport
from autodidact.cli import app

runner = CliRunner()


# ── Helpers ────────────────────────────────────────────────────────


def _make_query_response(**overrides) -> QueryResponse:
    defaults = dict(
        answer="Paris is the capital of France.",
        routed_to="local",
        confidence=0.92,
        cost_usd=0.0,
        learned=False,
        latency_ms=200,
    )
    defaults.update(overrides)
    return QueryResponse(**defaults)


def _make_savings_report(**overrides) -> SavingsReport:
    defaults = dict(
        total_queries=10,
        local_queries=7,
        cloud_queries=2,
        memory_queries=1,
        total_cost_usd=0.006,
        estimated_all_cloud_cost_usd=0.030,
        saved_usd=0.024,
        saved_pct=80.0,
        facts_learned=2,
    )
    defaults.update(overrides)
    return SavingsReport(**defaults)


# ── autodidact query ───────────────────────────────────────────────


class TestQueryCommand:
    """autodidact query 'question' — single query mode."""

    @patch("autodidact.cli.Agent")
    def test_query_prints_answer(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.query.return_value = _make_query_response()
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["query", "What is the capital of France?"])
        assert result.exit_code == 0
        assert "Paris" in result.output

    @patch("autodidact.cli.Agent")
    def test_query_passes_question_to_agent(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.query.return_value = _make_query_response()
        MockAgent.return_value = mock_agent

        runner.invoke(app, ["query", "What is the capital of France?"])
        mock_agent.query.assert_called_once()
        call_args = mock_agent.query.call_args
        assert call_args.args[0] == "What is the capital of France?"


# ── autodidact savings ─────────────────────────────────────────────


class TestSavingsCommand:
    """autodidact savings — cumulative cost savings."""

    @patch("autodidact.cli.Agent")
    def test_savings_shows_stats(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.savings.return_value = _make_savings_report()
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["savings"])
        assert result.exit_code == 0
        assert "10" in result.output  # total queries
        assert "7" in result.output   # local queries


# ── autodidact memory stats ────────────────────────────────────────


class TestMemoryStatsCommand:
    """autodidact memory stats — knowledge store info."""

    @patch("autodidact.cli.Agent")
    def test_memory_stats_shows_count(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.memory = MagicMock()
        mock_agent.memory.count.return_value = 42
        mock_agent.memory.get_stats.return_value = {"total": 42, "stm": 30, "ltm": 12}
        mock_agent.memory.list_domains.return_value = ["general", "science"]
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["memory", "stats"])
        assert result.exit_code == 0
        assert "42" in result.output


# ── autodidact memory search ──────────────────────────────────────


class TestMemorySearchCommand:
    """autodidact memory search 'query' — search learned knowledge."""

    @patch("autodidact.cli.Agent")
    def test_memory_search_no_results(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.memory = MagicMock()
        mock_agent._embed_client = MagicMock()
        mock_agent._embed_client.embed.return_value = [0.1] * 32
        mock_agent.memory.search.return_value = []
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["memory", "search", "quantum"])
        assert result.exit_code == 0
        assert "no" in result.output.lower() or "0" in result.output


# ── autodidact init ────────────────────────────────────────────────


class TestInitCommand:
    """autodidact init — interactive config generation (R8).

    These tests exercise the basic shape of the new zero-friction wizard.
    More detailed mode-specific tests live in test_init_wizard_integration.py.
    """

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_init_creates_config_file(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """local-only mode produces a config file with a 'local' section."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        config_path = tmp_path / "config.yaml"
        # mode=3 (local-only), local model default, db path
        input_text = "3\nqwen2.5:7b\n" + str(tmp_path / "memory.db") + "\n"

        result = runner.invoke(
            app, ["init", "--config-path", str(config_path)], input=input_text
        )
        assert result.exit_code == 0, result.output
        assert config_path.exists()

        config = yaml.safe_load(config_path.read_text())
        assert config["local"]["model"] == "qwen2.5:7b"

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_init_with_cloud_provider(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """local+cloud mode records the chosen cloud provider."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        config_path = tmp_path / "config.yaml"
        # mode=1, local model, cloud provider openai, api key, model default, db default
        input_text = "1\nqwen2.5:7b\nopenai\nsk-test123\n\n" + str(tmp_path / "memory.db") + "\n"

        result = runner.invoke(
            app, ["init", "--config-path", str(config_path)], input=input_text
        )
        assert result.exit_code == 0, result.output

        config = yaml.safe_load(config_path.read_text())
        assert config["cloud"]["provider"] == "openai"

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    @patch("autodidact.cli.detect_hardware")
    def test_init_defaults(self, mock_hw, mock_model, mock_detect, mock_smoke, tmp_path):
        """All defaults: mode=1 (local+cloud), default model, default cloud."""
        from autodidact.setup_wizard import OllamaStatus
        from autodidact.hardware import HardwareProfile
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        # Pin hardware tier so the recommended default is stable regardless
        # of the machine running the test.
        mock_hw.return_value = HardwareProfile(
            ram_gb=16, is_apple_silicon=False, has_nvidia=False,
            vram_gb=None, tier="medium",
        )
        config_path = tmp_path / "config.yaml"
        # mode default (1=local_cloud), model default, cloud=openai, api key, model default, db default
        input_text = "\n\nopenai\nsk-test\n\n\n"

        result = runner.invoke(
            app, ["init", "--config-path", str(config_path)], input=input_text
        )
        assert result.exit_code == 0, result.output

        config = yaml.safe_load(config_path.read_text())
        assert config["local"]["model"] == "qwen3:4b"
        assert "Ready" in result.output

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_init_writes_yaml_format(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """Config YAML has the expected top-level structure."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        config_path = tmp_path / "config.yaml"
        # mode=3 (local-only), defaults
        input_text = "3\n\n\n"

        runner.invoke(app, ["init", "--config-path", str(config_path)], input=input_text)

        raw = config_path.read_text()
        config = yaml.safe_load(raw)
        # Verify structure matches R8 spec
        assert "local" in config
        assert "routing" in config
        assert "memory" in config


# ── autodidact chat ────────────────────────────────────────────────


class TestChatCommand:
    """autodidact chat — interactive REPL."""

    @patch("autodidact.cli.Agent")
    def test_chat_exit_on_quit(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.savings.return_value = _make_savings_report()
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["chat"], input="quit\n")
        assert result.exit_code == 0

    @patch("autodidact.cli.Agent")
    def test_chat_processes_query(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.query.return_value = _make_query_response()
        mock_agent.savings.return_value = _make_savings_report()
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["chat"], input="What is Python?\nquit\n")
        assert result.exit_code == 0
        mock_agent.query.assert_called_once()
        call_args = mock_agent.query.call_args
        assert call_args.args[0] == "What is Python?"


# ── autodidact learn ──────────────────────────────────────────────


class TestLearnCommand:
    """autodidact learn <path> — document ingestion (R9)."""

    @patch("autodidact.cli.Agent")
    def test_learn_ingests_a_file(self, MockAgent, tmp_path):
        """`autodidact learn <file>` calls DocumentStore.ingest on that path."""
        mock_agent = MagicMock()
        mock_agent.documents = MagicMock()
        mock_result = MagicMock()
        mock_result.files_ingested = 1
        mock_result.chunks_created = 3
        mock_result.files_skipped = 0
        mock_agent.documents.ingest.return_value = mock_result
        MockAgent.return_value = mock_agent

        f = tmp_path / "note.md"
        f.write_text("A note.")

        result = runner.invoke(app, ["learn", str(f)])
        assert result.exit_code == 0, result.output
        mock_agent.documents.ingest.assert_called_once()
        # Output should include the counts.
        assert "1" in result.output
        assert "3" in result.output

    @patch("autodidact.cli.Agent")
    def test_learn_ingests_a_directory(self, MockAgent, tmp_path):
        """`autodidact learn <dir>` walks the directory."""
        mock_agent = MagicMock()
        mock_agent.documents = MagicMock()
        mock_result = MagicMock()
        mock_result.files_ingested = 5
        mock_result.chunks_created = 12
        mock_result.files_skipped = 0
        mock_agent.documents.ingest.return_value = mock_result
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["learn", str(tmp_path)])
        assert result.exit_code == 0
        mock_agent.documents.ingest.assert_called_once()

    @patch("autodidact.cli.Agent")
    def test_learn_stats_shows_totals(self, MockAgent):
        """`autodidact learn --stats` shows total files, chunks, sources."""
        mock_agent = MagicMock()
        mock_agent.documents = MagicMock()
        mock_agent.documents.get_stats.return_value = {
            "total_chunks": 42,
            "total_files": 7,
            "sources": {"/path/to/notes.md": 10, "/path/to/readme.md": 32},
        }
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["learn", "--stats"])
        assert result.exit_code == 0
        assert "42" in result.output  # total chunks
        assert "7" in result.output   # total files

    @patch("autodidact.cli.Agent")
    def test_learn_nonexistent_path_errors(self, MockAgent):
        """Ingesting a path that doesn't exist reports an error."""
        mock_agent = MagicMock()
        mock_agent.documents = MagicMock()
        MockAgent.return_value = mock_agent

        result = runner.invoke(app, ["learn", "/nonexistent/path/xyz"])
        assert result.exit_code != 0 or "not found" in result.output.lower() or "does not exist" in result.output.lower()
