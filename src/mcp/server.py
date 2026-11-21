"""Standalone stdio JSON-RPC 2.0 MCP Server backed by persistent SQLite WAL tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"

_FX_TO_CNY: dict[str, float] = {
    "CNY": 1.0000,
    "USD": 7.1850,
    "HKD": 0.9230,
    "EUR": 7.7600,
    "SGD": 5.3800,
    "JPY": 0.0475,
}

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "mcp_itinerary_query",
        "description": "[MCP Calendar] Query persistent travel itinerary legs and scheduled meetings.",
        "inputSchema": {
            "type": "object",
            "properties": {"date_prefix": {"type": "string", "default": "2026-11"}},
        },
        "annotations": {
            "namespace": "mcp_calendar",
            "pair_group": "mcp_itinerary",
            "keywords": ["日程", "日历", "排期", "会议", "冲突", "时间段", "calendar", "itinerary"],
        },
    },
    {
        "name": "mcp_itinerary_schedule",
        "description": "[MCP Calendar] Schedule and persist a new travel review or meeting slot after conflict check.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 2},
                "slot_time": {"type": "string", "minLength": 6},
                "location": {"type": "string", "default": "Online"},
            },
            "required": ["title", "slot_time"],
        },
        "annotations": {
            "namespace": "mcp_calendar",
            "pair_group": "mcp_itinerary",
            "keywords": ["锁定日程", "写入日程", "创建日程", "预约", "会议", "schedule"],
        },
    },
    {
        "name": "mcp_expense_pool_query",
        "description": "[MCP Finance] Query project remaining budget pool, approver, and lodging cap from SQLite.",
        "inputSchema": {
            "type": "object",
            "properties": {"project_code": {"type": "string", "default": "PRJ-PANGU-2"}},
        },
        "annotations": {
            "namespace": "mcp_finance",
            "pair_group": "mcp_expense",
            "keywords": ["预算池", "剩余额度", "经费", "项目代码", "盘古", "星海", "budget", "voucher"],
        },
    },
    {
        "name": "mcp_expense_export_voucher",
        "description": "[MCP Finance] Export a structured JSON reimbursement voucher bound to a project code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "voucher_name": {"type": "string"},
                "project_code": {"type": "string"},
                "net_reimbursable_cny": {"type": "number", "minimum": 0},
                "notes": {"type": "string", "default": ""},
            },
            "required": ["voucher_name", "project_code", "net_reimbursable_cny"],
        },
        "annotations": {
            "namespace": "mcp_finance",
            "pair_group": "mcp_expense",
            "keywords": ["凭证", "核销单", "voucher", ".json", "财务导出"],
        },
    },
    {
        "name": "mcp_travel_notes_search",
        "description": "[MCP Notes] Search persistent personal travel checklists and invoice reminders.",
        "inputSchema": {
            "type": "object",
            "properties": {"keyword": {"type": "string", "default": ""}},
        },
        "annotations": {
            "namespace": "mcp_notes",
            "pair_group": "mcp_notes",
            "keywords": ["备忘录", "笔记", "清单", "注意事项", "notes", "checklist"],
        },
    },
    {
        "name": "mcp_travel_notes_upsert",
        "description": "[MCP Notes] Create or update a persistent personal travel checklist note.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_key": {"type": "string", "minLength": 2},
                "content": {"type": "string", "minLength": 3},
            },
            "required": ["note_key", "content"],
        },
        "annotations": {
            "namespace": "mcp_notes",
            "pair_group": "mcp_notes",
            "keywords": ["更新备忘录", "写入备忘录", "差旅笔记", "清单更新", "upsert"],
        },
    },
    {
        "name": "mcp_policy_lookup",
        "description": "[MCP Policy] Look up standard corporate hotel and meal per-diem caps by city tier.",
        "inputSchema": {
            "type": "object",
            "properties": {"city_tier": {"type": "string", "default": "tier_1"}},
        },
        "annotations": {
            "namespace": "mcp_policy",
            "pair_group": "mcp_payout",
            "keywords": ["差旅制度", "标准差标", "一线城市", "补贴", "policy", "打款"],
        },
    },
    {
        "name": "mcp_reimburse_submit_payout",
        "description": "[MCP Policy] Submit an idempotent direct reimbursement payout request to corporate finance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_code": {"type": "string"},
                "amount_cny": {"type": "number", "exclusiveMinimum": 0},
                "payee_note": {"type": "string", "default": ""},
                "idempotency_key": {"type": "string", "default": ""},
            },
            "required": ["project_code", "amount_cny"],
        },
        "annotations": {
            "namespace": "mcp_policy",
            "pair_group": "mcp_payout",
            "keywords": ["直接发起打款", "立即打款", "自动打款", "转账", "payout", "reimburse_submit"],
        },
    },
    {
        "name": "mcp_per_diem_currency_rate",
        "description": "[MCP FX] Query official corporate reimbursement FX rate to CNY.",
        "inputSchema": {
            "type": "object",
            "properties": {"currency": {"type": "string"}},
            "required": ["currency"],
        },
        "annotations": {
            "namespace": "mcp_fx",
            "keywords": ["汇率", "美元", "港币", "日元", "usd", "hkd", "jpy", "eur", "sgd", "fx"],
        },
    },
]


class StewardMcpBackend:
    """SQLite WAL persistence engine backing all MCP domain tools."""

    def __init__(self, db_path: Path, workspace_dir: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = workspace_dir.resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.lock_flag_path = self.db_path.with_suffix(".lock_once")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS itinerary_events (
                slot_time TEXT PRIMARY KEY, title TEXT NOT NULL, location TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS budget_pools (
                project_code TEXT PRIMARY KEY, project_name TEXT NOT NULL,
                approver TEXT NOT NULL, remaining_budget_cny REAL NOT NULL, tier1_hotel_cap_cny REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS travel_notes (
                note_key TEXT PRIMARY KEY, content TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS payout_ledger (
                payout_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE,
                project_code TEXT NOT NULL, amount_cny REAL NOT NULL, payee_note TEXT NOT NULL
            );
            """
        )
        cur.executemany(
            "INSERT OR IGNORE INTO itinerary_events(slot_time, title, location) VALUES (?, ?, ?)",
            [
                ("2026-11-19 10:00", "盘古二期客户技术对齐", "深圳南山科技园 T3"),
                ("2026-11-20 10:30", "AI基础设施双周同步会", "线上会议室 B"),
            ],
        )
        cur.executemany(
            "INSERT OR IGNORE INTO budget_pools VALUES (?, ?, ?, ?, ?)",
            [
                ("PRJ-PANGU-2", "盘古二期项目", "陈总监 (Chen Director)", 48500.0, 850.0),
                ("PRJ-XINGHAI-1", "星海一期项目", "林经理 (Lin Manager)", 22000.0, 650.0),
                ("PRJ-KUNLUN-3", "昆仑底座项目", "赵总监 (Zhao Director)", 31000.0, 700.0),
            ],
        )
        cur.executemany(
            "INSERT OR IGNORE INTO travel_notes VALUES (?, ?)",
            [
                ("shenzhen_packing", "携带身份证原件、保密协议签署件、Type-C投屏转换器；返程前收集酒店增值税专票与水单。"),
                ("reimburse_checklist", "住宿报销需同时附发票与每日明细水单（剔除迷你吧、洗衣等个人消费）。"),
            ],
        )
        self.conn.commit()

    def _safe_path(self, filename: str) -> Path:
        clean = Path(filename).name
        if not clean or clean in {".", ".."} or clean.startswith(".env"):
            raise ValueError(f"Disallowed filename: {filename!r}")
        resolved = (self.workspace_dir / clean).resolve()
        if self.workspace_dir not in resolved.parents and resolved != self.workspace_dir:
            raise ValueError(f"Path escapes workspace: {filename!r}")
        return resolved

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        cur = self.conn.cursor()
        if name == "mcp_itinerary_query":
            prefix = str(args.get("date_prefix", "2026-11"))
            rows = cur.execute(
                "SELECT slot_time, title, location FROM itinerary_events WHERE slot_time LIKE ? ORDER BY slot_time",
                (f"{prefix}%",),
            ).fetchall()
            evs = [dict(r) for r in rows]
            return {"status": "ok", "events": evs, "count": len(evs)}

        if name == "mcp_itinerary_schedule":
            slot = str(args["slot_time"])
            title = str(args["title"])
            loc = str(args.get("location", "Online"))
            exists = cur.execute("SELECT title FROM itinerary_events WHERE slot_time = ?", (slot,)).fetchone()
            if exists:
                return {"status": "error", "error": f"Time slot {slot} conflicts with {exists['title']}"}
            cur.execute("INSERT INTO itinerary_events(slot_time, title, location) VALUES (?, ?, ?)", (slot, title, loc))
            self.conn.commit()
            return {"status": "ok", "scheduled": {"slot_time": slot, "title": title, "location": loc}}

        if name == "mcp_expense_pool_query":
            q = str(args.get("project_code", "")).strip()
            row = cur.execute(
                "SELECT * FROM budget_pools WHERE UPPER(project_code) = UPPER(?) OR project_name = ?",
                (q, q),
            ).fetchone()
            if row:
                return {"status": "ok", **dict(row)}
            return {"status": "ok", "pools": [dict(r) for r in cur.execute("SELECT * FROM budget_pools").fetchall()]}

        if name == "mcp_expense_export_voucher":
            vname = str(args["voucher_name"])
            if not vname.endswith(".json"):
                vname += ".json"
            target = self._safe_path(vname)
            payload = {
                "project_code": str(args["project_code"]),
                "net_reimbursable_cny": round(float(args["net_reimbursable_cny"]), 2),
                "notes": str(args.get("notes", "")),
            }
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"status": "ok", "voucher_file": target.name, "payload": payload}

        if name == "mcp_travel_notes_search":
            kw = str(args.get("keyword", "")).strip().lower()
            rows = cur.execute("SELECT note_key, content FROM travel_notes").fetchall()
            hits = {r["note_key"]: r["content"] for r in rows if not kw or kw in r["note_key"].lower() or kw in r["content"].lower()}
            return {"status": "ok", "notes": hits or {r["note_key"]: r["content"] for r in rows}}

        if name == "mcp_travel_notes_upsert":
            if self.lock_flag_path.exists():
                self.lock_flag_path.unlink(missing_ok=True)
                return {"status": "error", "error": "SQLite BusyError: travel_notes store temporarily locked; retry mcp_travel_notes_upsert."}
            key, body = str(args["note_key"]), str(args["content"])
            cur.execute("INSERT OR REPLACE INTO travel_notes(note_key, content) VALUES (?, ?)", (key, body))
            self.conn.commit()
            return {"status": "ok", "note_key": key, "saved_length": len(body)}

        if name == "mcp_policy_lookup":
            tier = str(args.get("city_tier", "tier_1")).lower()
            is_t1 = ("1" in tier) or ("一线" in tier)
            return {
                "status": "ok",
                "city_tier": "tier_1" if is_t1 else "tier_2",
                "standard_hotel_cap_cny": 700.0 if is_t1 else 500.0,
                "meal_per_diem_cny": 120.0 if is_t1 else 100.0,
                "personal_exclusions": ["迷你吧/Minibar", "私人洗衣", "付费影视"],
            }

        if name == "mcp_reimburse_submit_payout":
            amt = round(float(args["amount_cny"]), 2)
            if amt <= 0:
                return {"status": "error", "error": "amount_cny must be strictly positive"}
            idem = str(args.get("idempotency_key", "")).strip() or None
            if idem:
                prev = cur.execute("SELECT * FROM payout_ledger WHERE idempotency_key = ?", (idem,)).fetchone()
                if prev:
                    return {"status": "ok", "idempotent_replay": True, "submitted_payout": dict(prev)}
            cnt = int(cur.execute("SELECT COUNT(*) FROM payout_ledger").fetchone()[0])
            pid = f"PAY-{cnt + 1:04d}"
            cur.execute(
                "INSERT INTO payout_ledger(payout_id, idempotency_key, project_code, amount_cny, payee_note) VALUES (?, ?, ?, ?, ?)",
                (pid, idem, str(args["project_code"]), amt, str(args.get("payee_note", ""))),
            )
            self.conn.commit()
            return {
                "status": "ok",
                "idempotent_replay": False,
                "submitted_payout": {
                    "payout_id": pid,
                    "project_code": str(args["project_code"]),
                    "amount_cny": amt,
                    "payee_note": str(args.get("payee_note", "")),
                },
            }

        if name == "mcp_per_diem_currency_rate":
            code = str(args["currency"]).strip().upper()
            if code not in _FX_TO_CNY:
                return {"status": "error", "error": f"Unsupported currency code: {code!r}"}
            return {"status": "ok", "currency": code, "rate_to_cny": _FX_TO_CNY[code]}

        return {"status": "error", "error": f"Unknown MCP tool: {name!r}"}


def run_stdio_server(db_path: Path, workspace_dir: Path) -> None:
    backend = StewardMcpBackend(db_path, workspace_dir)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        if method == "notifications/initialized":
            continue
        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "tripleger-steward-mcp", "version": "2.0.0"},
                },
            }
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFS}}
        elif method == "tools/call":
            tname = str(params.get("name", ""))
            targs = params.get("arguments") or {}
            try:
                out = backend.dispatch(tname, targs)
                is_err = out.get("status") != "ok"
            except Exception as exc:
                out = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                is_err = True
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                    "structuredContent": out,
                    "isError": is_err,
                },
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--workspace", required=True)
    ns = p.parse_args()
    run_stdio_server(Path(ns.db), Path(ns.workspace))
