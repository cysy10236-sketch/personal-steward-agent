"""Stdio JSON-RPC 2.0 MCP client with sanitized child env and conservative crash recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from ..router.steward_router import ToolSchemaSpec


class StdioMcpClient:
    """Manages an isolated MCP server subprocess over newline-delimited JSON-RPC 2.0 stdio."""

    def __init__(self, db_path: Path | str, workspace_dir: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[str] | None = None
        self._req_id = 0
        self._lock = threading.Lock()
        self._discovered_specs: list[ToolSchemaSpec] = []
        self._start_and_handshake()

    @staticmethod
    def _sanitized_env() -> dict[str, str]:
        """Strip host API keys & secrets before spawning external MCP server subprocess."""
        allowed = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONUTF8", "PYTHONPATH")
        env = {k: os.environ[k] for k in allowed if k in os.environ}
        env["PYTHONUTF8"] = "1"
        return env

    def _start_and_handshake(self) -> None:
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.mcp.server",
                "--db",
                str(self.db_path),
                "--workspace",
                str(self.workspace_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=self._sanitized_env(),
        )
        init_res = self._rpc("initialize", {"protocolVersion": "2024-11-05", "clientInfo": {"name": "steward"}})
        if "protocolVersion" not in init_res:
            raise RuntimeError(f"MCP initialize handshake failed: {init_res}")
        self._notify("notifications/initialized", {})
        tools_res = self._rpc("tools/list", {})
        specs: list[ToolSchemaSpec] = []
        for t in tools_res.get("tools", []):
            ann = t.get("annotations") or {}
            specs.append(
                ToolSchemaSpec(
                    name=str(t["name"]),
                    description=str(t.get("description", "")),
                    parameters=dict(t.get("inputSchema") or {"type": "object", "properties": {}}),
                    namespace=str(ann.get("namespace", "mcp")),
                    pair_group=ann.get("pair_group"),
                    always_include=False,
                    keywords=list(ann.get("keywords", [])),
                )
            )
        self._discovered_specs = specs

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self._proc.stdin.flush()

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._req_id += 1
        req_id = self._req_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise BrokenPipeError("MCP server terminated unexpectedly before replying")
        msg = json.loads(line)
        if "error" in msg:
            raise RuntimeError(f"MCP JSON-RPC error: {msg['error']}")
        return dict(msg.get("result") or {})

    @property
    def tool_specs(self) -> list[ToolSchemaSpec]:
        return list(self._discovered_specs)

    def arm_transient_notes_lock_once(self) -> None:
        self.db_path.with_suffix(".lock_once").write_text("1", encoding="utf-8")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke an MCP tool; restarts dead server for future calls without replaying uncertain writes."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start_and_handshake()
            try:
                res = self._rpc("tools/call", {"name": name, "arguments": arguments})
            except BrokenPipeError:
                self.close()
                self._start_and_handshake()
                return {
                    "status": "error",
                    "error": f"MCP server crashed during {name}; server restarted without auto-replaying uncertain side effect.",
                }
            if isinstance(res.get("structuredContent"), dict):
                return dict(res["structuredContent"])
            content = res.get("content") or []
            if content and isinstance(content[0], dict):
                try:
                    return json.loads(str(content[0].get("text", "{}")))
                except json.JSONDecodeError:
                    return {"status": "ok", "text": str(content[0].get("text", ""))}
            return {"status": "error", "error": "Empty MCP response"}

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3.0)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None
