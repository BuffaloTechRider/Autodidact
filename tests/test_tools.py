"""Tests for the v2 tool foundation: registry, terminal, and file_ops.

The registry tests use isolated ``ToolRegistry`` instances so they don't
depend on or pollute the shared ``REGISTRY`` singleton. The tool tests
exercise the real handlers against ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from autodidact.tools import REGISTRY
from autodidact.tools.registry import ToolRegistry
from autodidact.tools import file_ops, terminal


# ── Registry ─────────────────────────────────────────────────────


class TestRegistry:
    def _reg(self):
        r = ToolRegistry()
        r.register(
            "echo",
            description="echo back the message",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "times": {"type": "integer"},
                },
                "required": ["message"],
            },
            handler=lambda a: {"said": a["message"] * a.get("times", 1)},
            toolset="test",
        )
        return r

    def test_schema_is_openai_function_format(self):
        r = self._reg()
        schema = r.get_schemas()[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "parameters" in schema["function"]

    def test_get_schemas_filters_by_toolset(self):
        r = self._reg()
        r.register(
            "other",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda a: None,
            toolset="misc",
        )
        assert len(r.get_schemas(["test"])) == 1
        assert len(r.get_schemas(["test", "misc"])) == 2
        assert len(r.get_schemas()) == 2

    def test_dispatch_success_envelope(self):
        r = self._reg()
        out = json.loads(r.dispatch("echo", {"message": "hi"}))
        assert out == {"ok": True, "result": {"said": "hi"}}

    def test_dispatch_unknown_tool(self):
        r = self._reg()
        out = json.loads(r.dispatch("nope", {}))
        assert out["ok"] is False
        assert "unknown tool" in out["error"]

    def test_dispatch_handler_exception_is_captured(self):
        r = ToolRegistry()
        r.register(
            "boom",
            description="raises",
            parameters={"type": "object", "properties": {}},
            handler=lambda a: (_ for _ in ()).throw(RuntimeError("kaboom")),
        )
        out = json.loads(r.dispatch("boom", {}))
        assert out["ok"] is False
        assert "kaboom" in out["error"]

    def test_argument_type_coercion(self):
        r = self._reg()
        # "times" declared integer; model sends a string.
        out = json.loads(r.dispatch("echo", {"message": "ab", "times": "3"}))
        assert out["result"]["said"] == "ababab"

    def test_reregister_replaces_without_error(self):
        r = ToolRegistry()
        r.register(
            "t", description="v1", parameters={"type": "object", "properties": {}},
            handler=lambda a: 1,
        )
        r.register(
            "t", description="v2", parameters={"type": "object", "properties": {}},
            handler=lambda a: 2,
        )
        assert json.loads(r.dispatch("t", {}))["result"] == 2

    def test_shared_registry_has_expected_tools(self):
        names = set(REGISTRY.names())
        assert {"terminal", "read_file", "write_file", "edit_file",
                "search_files", "list_directory"} <= names


# ── Terminal ─────────────────────────────────────────────────────


class TestTerminal:
    def test_runs_command_and_captures_output(self):
        out = terminal.run_terminal({"command": "echo hello"})
        assert out["exit_code"] == 0
        assert out["output"].strip() == "hello"
        assert out["timed_out"] is False

    def test_nonzero_exit_is_normal_result(self):
        out = terminal.run_terminal({"command": "exit 3"})
        assert out["exit_code"] == 3
        assert out["timed_out"] is False

    def test_captures_stderr(self):
        out = terminal.run_terminal({"command": "echo oops 1>&2"})
        assert "oops" in out["output"]

    def test_timeout_reported_not_raised(self):
        out = terminal.run_terminal({"command": "sleep 5", "timeout": 1})
        assert out["timed_out"] is True
        assert out["exit_code"] is None

    def test_empty_command_raises(self):
        with pytest.raises(ValueError):
            terminal.run_terminal({"command": ""})

    def test_output_truncation(self):
        # Emit more than the byte budget; result must be capped with a marker.
        big = terminal.run_terminal(
            {"command": "python -c \"print('x'*40000)\""}
        )
        assert "truncated" in big["output"]
        assert len(big["output"].encode()) <= terminal._MAX_OUTPUT_BYTES + 200


# ── File ops ─────────────────────────────────────────────────────


class TestFileOps:
    @pytest.fixture(autouse=True)
    def _in_tmp_cwd(self, tmp_path, monkeypatch):
        """Run each file-ops test inside tmp_path.

        The tools confine paths to the current working directory, so tests
        operate relative to a temp cwd (mirroring the agent running inside its
        project directory). Paths below are given relative to tmp_path.
        """
        monkeypatch.chdir(tmp_path)

    def test_write_then_read_roundtrip(self, tmp_path):
        p = tmp_path / "sub" / "note.txt"
        w = file_ops.write_file({"path": str(p), "content": "hello world"})
        assert w["bytes_written"] == 11
        r = file_ops.read_file({"path": str(p)})
        assert r["content"] == "hello world"
        assert r["truncated"] is False

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            file_ops.read_file({"path": str(tmp_path / "nope.txt")})

    def test_edit_unique_substring(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("alpha beta gamma")
        file_ops.edit_file({"path": str(p), "old": "beta", "new": "BETA"})
        assert p.read_text() == "alpha BETA gamma"

    def test_edit_missing_substring_raises(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("alpha")
        with pytest.raises(ValueError, match="not found"):
            file_ops.edit_file({"path": str(p), "old": "zzz", "new": "y"})

    def test_edit_non_unique_substring_raises(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x x x")
        with pytest.raises(ValueError, match="multiple"):
            file_ops.edit_file({"path": str(p), "old": "x", "new": "y"})

    def test_edit_fuzzy_matches_indentation_drift(self, tmp_path):
        # File block is indented 8 spaces; the model's multi-line 'old' is
        # de-indented. Line-trimmed fuzzy matching finds it, and the new text
        # is re-anchored to the file's actual indentation.
        p = tmp_path / "code.py"
        p.write_text("def f():\n        a = 1\n        b = 2\n")
        res = file_ops.edit_file(
            {"path": str(p), "old": "a = 1\nb = 2", "new": "a = 10\nb = 20"}
        )
        assert res["strategy"] == "line-trimmed"
        assert p.read_text() == "def f():\n        a = 10\n        b = 20\n"

    def test_edit_exact_match_takes_priority(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("alpha beta")
        res = file_ops.edit_file({"path": str(p), "old": "beta", "new": "B"})
        assert res["strategy"] == "exact"
        assert p.read_text() == "alpha B"

    def test_path_escape_is_rejected(self, tmp_path):
        # A traversal outside the working directory must be refused.
        with pytest.raises(PermissionError, match="escapes"):
            file_ops.read_file({"path": "../../../etc/passwd"})

    def test_search_finds_matches(self, tmp_path):
        (tmp_path / "a.txt").write_text("foo\nbar\nfoobar")
        (tmp_path / "b.txt").write_text("nothing here")
        res = file_ops.search_files({"pattern": r"foo", "path": str(tmp_path)})
        assert res["count"] == 2
        assert all("foo" in m["text"] for m in res["matches"])

    def test_search_invalid_regex_raises(self, tmp_path):
        with pytest.raises(ValueError, match="invalid regex"):
            file_ops.search_files({"pattern": "(", "path": str(tmp_path)})

    def test_search_truncates_at_cap(self, tmp_path):
        # More matching lines than the cap → truncated flag set.
        lines = "\n".join(f"match {i}" for i in range(file_ops._MAX_SEARCH_MATCHES + 20))
        (tmp_path / "big.txt").write_text(lines)
        res = file_ops.search_files({"pattern": "match", "path": str(tmp_path)})
        assert res["truncated"] is True
        assert res["count"] == file_ops._MAX_SEARCH_MATCHES

    def test_list_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        res = file_ops.list_directory({"path": str(tmp_path)})
        names = {e["name"]: e["is_dir"] for e in res["entries"]}
        assert names == {"file.txt": False, "subdir": True}

    def test_list_non_directory_raises(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x")
        with pytest.raises(NotADirectoryError):
            file_ops.list_directory({"path": str(p)})
