"""Real stdio JSON-RPC 2.0 MCP server and client with SQLite WAL persistence."""

from .client import StdioMcpClient

__all__ = ["StdioMcpClient"]
