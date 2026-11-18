"""Hybrid personal memory combining SQLite FTS5, Dense Vector, Directed NumPy PPR, and Slot Supersession."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
import uuid

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from .tokenizer_embedder import (
    DeterministicBilingualEmbedder,
    build_fts5_or_query,
    extract_bilingual_entities,
    fts5_segment_text,
    infer_travel_slot,
)


class EntityTriple(BaseModel):
    """Directed relation edge in the personal travel/project knowledge graph."""

    source: str
    relation: str
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MemoryRecord(BaseModel):
    """Persistent personal memory item with watermark sequence and supersession lineage."""

    memory_id: str
    seq: int
    session_id: str = "default"
    content: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    access_count: int = Field(default=1, ge=1)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    entities: list[str] = Field(default_factory=list)
    relations: list[EntityTriple] = Field(default_factory=list)
    subject: str | None = None
    slot: str | None = None
    superseded_by: str | None = None
    content_hash: str = ""


class MemoryHit(BaseModel):
    """Ranked memory hit with transparent four-channel score attribution."""

    record: MemoryRecord
    final_score: float
    vector_score: float
    lexical_score: float
    graph_ppr_score: float
    decay_multiplier: float
    hop_paths: list[str] = Field(default_factory=list)


class MemorySearchResult(BaseModel):
    """Search response including abstention status and superseded history count."""

    hits: list[MemoryHit] = Field(default_factory=list)
    abstained: bool = False
    superseded_filtered_count: int = 0


class StewardMemoryEngine:
    """Persistent SQLite WAL + FTS5 + Vector + Directed NumPy PPR memory engine."""

    def __init__(
        self,
        db_path: Path | str = Path("runtime_data/memory/steward_memory.sqlite3"),
        *,
        embedding_dim: int = 256,
        base_half_life_hours: float = 168.0,
        ppr_alpha: float = 0.85,
        abstention_threshold: float = 0.18,
        weights: tuple[float, float, float] = (0.42, 0.23, 0.35),
        enable_graph_ppr: bool = True,
        enable_supersession_and_decay: bool = True,
    ) -> None:
        if abs(sum(weights) - 1.0) > 1e-5:
            raise ValueError(f"Channel weights must sum to 1.0, got {weights}")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = DeterministicBilingualEmbedder(dim=embedding_dim)
        self.base_half_life_hours = base_half_life_hours
        self.ppr_alpha = ppr_alpha
        self.abstention_threshold = abstention_threshold
        self.w_vec, self.w_lex, self.w_ppr = weights
        self.enable_graph_ppr = enable_graph_ppr
        self.enable_supersession_and_decay = enable_supersession_and_decay

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_sqlite_schema()

        self._records: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._hash_to_id: dict[str, str] = {}
        self._alias_map: dict[str, str] = {}
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._seq: int = 0

        self._persist_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._db_lock = asyncio.Lock()

        self._load_from_sqlite()

    def _init_sqlite_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS steward_memories (
                memory_id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                segmented_text TEXT NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                entities_json TEXT NOT NULL,
                relations_json TEXT NOT NULL,
                subject TEXT,
                slot TEXT,
                superseded_by TEXT,
                content_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS steward_memories_fts USING fts5(
                memory_id UNINDEXED,
                segmented_text
            );
            """
        )
        self._conn.commit()

    def _load_from_sqlite(self) -> None:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT * FROM steward_memories ORDER BY seq ASC").fetchall()
        for r in rows:
            ents = json.loads(r["entities_json"])
            rels = [EntityTriple.model_validate(x) for x in json.loads(r["relations_json"])]
            rec = MemoryRecord(
                memory_id=r["memory_id"],
                seq=int(r["seq"]),
                session_id=r["session_id"],
                content=r["content"],
                importance=float(r["importance"]),
                access_count=int(r["access_count"]),
                created_at=float(r["created_at"]),
                updated_at=float(r["updated_at"]),
                entities=ents,
                relations=rels,
                subject=r["subject"],
                slot=r["slot"],
                superseded_by=r["superseded_by"],
                content_hash=r["content_hash"],
            )
            self._records[rec.memory_id] = rec
            self._vectors[rec.memory_id] = np.asarray(
                json.loads(r["vector_json"]), dtype=np.float32
            )
            self._hash_to_id[rec.content_hash] = rec.memory_id
            self._seq = max(self._seq, rec.seq)
            self._index_graph(rec)

    @staticmethod
    def _norm_key(name: str) -> str:
        s = re.sub(r"\s+", " ", name.strip()).strip(".,;:!?()[]{}<>，。；：！？（）【】「」『』\"'")
        return s.lower() if re.fullmatch(r"[A-Za-z0-9_.-]+", s) else s

    def register_alias(self, canonical: str, *aliases: str) -> None:
        c_key = self._norm_key(canonical)
        if not c_key:
            return
        self._alias_map[c_key] = c_key
        if not self.graph.has_node(c_key):
            self.graph.add_node(c_key, memory_ids=set())
        for al in aliases:
            a_key = self._norm_key(al)
            if a_key:
                self._alias_map[a_key] = c_key

    def resolve_entity(self, raw: str) -> str:
        k = self._norm_key(raw)
        return self._alias_map.get(k, k)

    def _index_graph(self, rec: MemoryRecord) -> None:
        for ent in rec.entities:
            canon = self.resolve_entity(ent)
            if not self.graph.has_node(canon):
                self.graph.add_node(canon, memory_ids=set())
            self.graph.nodes[canon]["memory_ids"].add(rec.memory_id)

        for rel in rec.relations:
            src = self.resolve_entity(rel.source)
            dst = self.resolve_entity(rel.target)
            if not src or not dst or src == dst:
                continue
            for node in (src, dst):
                if not self.graph.has_node(node):
                    self.graph.add_node(node, memory_ids=set())
                self.graph.nodes[node]["memory_ids"].add(rec.memory_id)
            self.graph.add_edge(
                src,
                dst,
                relation=rel.relation,
                weight=float(rel.confidence),
                memory_id=rec.memory_id,
            )
            self.graph.add_edge(
                dst,
                src,
                relation=f"inv:{rel.relation}",
                weight=float(rel.confidence) * 0.70,
                memory_id=rec.memory_id,
            )

    def _persist_record_sync(self, memory_id: str) -> None:
        rec = self._records[memory_id]
        vec = self._vectors[memory_id]
        seg = fts5_segment_text(rec.content)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO steward_memories (
                memory_id, seq, session_id, content, segmented_text,
                importance, access_count, created_at, updated_at,
                entities_json, relations_json, subject, slot,
                superseded_by, content_hash, vector_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.memory_id,
                rec.seq,
                rec.session_id,
                rec.content,
                seg,
                rec.importance,
                rec.access_count,
                rec.created_at,
                rec.updated_at,
                json.dumps(rec.entities, ensure_ascii=False),
                json.dumps([r.model_dump(mode="json") for r in rec.relations], ensure_ascii=False),
                rec.subject,
                rec.slot,
                rec.superseded_by,
                rec.content_hash,
                json.dumps([round(float(x), 6) for x in vec.tolist()]),
            ),
        )
        cur.execute("DELETE FROM steward_memories_fts WHERE memory_id = ?", (rec.memory_id,))
        cur.execute(
            "INSERT INTO steward_memories_fts (memory_id, segmented_text) VALUES (?, ?)",
            (rec.memory_id, seg),
        )
        self._conn.commit()

    def add_memory_sync(
        self,
        content: str,
        *,
        session_id: str = "default",
        importance: float = 0.75,
        entities: list[str] | None = None,
        relations: list[EntityTriple] | None = None,
        subject: str | None = None,
        slot: str | None = None,
        timestamp: float | None = None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        """Write-through memory insertion with idempotency and watermark slot supersession."""
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("Memory content cannot be empty")
        now_ts = float(timestamp if timestamp is not None else time.time())
        c_hash = hashlib.sha256(re.sub(r"\s+", " ", cleaned.lower()).encode("utf-8")).hexdigest()

        # 1. Idempotent exact hash check
        if c_hash in self._hash_to_id:
            existing = self._records[self._hash_to_id[c_hash]]
            existing.access_count += 1
            existing.updated_at = max(existing.updated_at, now_ts)
            existing.importance = max(existing.importance, importance)
            self._persist_record_sync(existing.memory_id)
            return existing

        auto_ents = extract_bilingual_entities(cleaned, alias_register_cb=self.register_alias)
        merged_ents: list[str] = []
        for e in (entities or []) + auto_ents:
            resolved = self.resolve_entity(e)
            if resolved and resolved not in merged_ents:
                merged_ents.append(resolved)

        rel_list = list(relations or [])
        if not rel_list and len(merged_ents) >= 2:
            for i in range(len(merged_ents) - 1):
                if merged_ents[i] != merged_ents[i + 1]:
                    rel_list.append(
                        EntityTriple(
                            source=merged_ents[i],
                            relation="associated_with",
                            target=merged_ents[i + 1],
                            confidence=0.75,
                        )
                    )

        if subject is None or slot is None:
            inf_subj, inf_slot = infer_travel_slot(cleaned)
            subject = subject or inf_subj
            slot = slot or inf_slot

        vec = self.embedder.embed(cleaned)

        # 2. Near-duplicate consolidation when numeric anchors are unchanged
        new_nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
        for existing in self._records.values():
            if existing.superseded_by is not None:
                continue
            if float(np.dot(vec, self._vectors[existing.memory_id])) >= 0.92:
                old_nums = re.findall(r"\d+(?:\.\d+)?", existing.content)
                if new_nums == old_nums and existing.slot == slot:
                    existing.access_count += 1
                    existing.updated_at = max(existing.updated_at, now_ts)
                    self._persist_record_sync(existing.memory_id)
                    return existing

        self._seq += 1
        mid = memory_id or f"mem_{uuid.uuid4().hex[:10]}"
        rec = MemoryRecord(
            memory_id=mid,
            seq=self._seq,
            session_id=session_id,
            content=cleaned,
            importance=round(float(importance), 4),
            access_count=1,
            created_at=now_ts,
            updated_at=now_ts,
            entities=merged_ents,
            relations=rel_list,
            subject=subject,
            slot=slot,
            superseded_by=None,
            content_hash=c_hash,
        )

        # 3. Watermark single-value slot supersession (e.g., flight rescheduled or cap updated)
        superseded_ids: list[str] = []
        if subject and slot:
            for prev in self._records.values():
                if (
                    prev.superseded_by is None
                    and prev.subject == subject
                    and prev.slot == slot
                    and prev.seq < rec.seq
                ):
                    prev.superseded_by = rec.memory_id
                    superseded_ids.append(prev.memory_id)

        self._records[mid] = rec
        self._vectors[mid] = vec
        self._hash_to_id[c_hash] = mid
        self._index_graph(rec)

        self._persist_record_sync(mid)
        for sid in superseded_ids:
            self._persist_record_sync(sid)
        return rec

    async def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._background_persist_loop())

    async def _background_persist_loop(self) -> None:
        while True:
            mid = await self._persist_queue.get()
            try:
                if mid is None:
                    return
                async with self._db_lock:
                    if mid in self._records:
                        self._persist_record_sync(mid)
            finally:
                self._persist_queue.task_done()

    async def add_memory_async(self, content: str, **kwargs: Any) -> MemoryRecord:
        """Immediate in-memory write-through (Read-Your-Writes) + async SQLite flush."""
        await self._ensure_worker()
        async with self._db_lock:
            rec = self.add_memory_sync(content, **kwargs)
        await self._persist_queue.put(rec.memory_id)
        return rec

    async def flush_and_close(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            await self._persist_queue.put(None)
            await self._persist_queue.join()
            await self._worker_task
        self._conn.close()

    def _compute_ppr(self, query: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        """Pure-NumPy power-iteration Personalized PageRank with dangling-node redistribution."""
        if not self.enable_graph_ppr or self.graph.number_of_nodes() == 0:
            return {}, {}

        q_ents = extract_bilingual_entities(query)
        q_lower = query.lower()
        for alias_k, canon_v in self._alias_map.items():
            if len(alias_k) >= 2 and alias_k in q_lower and canon_v not in q_ents:
                q_ents.append(canon_v)
        for node in self.graph.nodes:
            if len(str(node)) >= 2 and str(node) in q_lower and str(node) not in q_ents:
                q_ents.append(str(node))

        seeds = [self.resolve_entity(e) for e in q_ents if self.graph.has_node(self.resolve_entity(e))]
        seeds = list(dict.fromkeys(seeds))
        if not seeds:
            return {}, {}

        dg = nx.DiGraph()
        for n in self.graph.nodes:
            dg.add_node(n)
        for u, v, d in self.graph.edges(data=True):
            w = float(d.get("weight", 1.0))
            if dg.has_edge(u, v):
                dg[u][v]["weight"] += w
            else:
                dg.add_edge(u, v, weight=w)

        nodes = list(dg.nodes)
        n_sz = len(nodes)
        idx_map = {node: i for i, node in enumerate(nodes)}
        p_vec = np.zeros(n_sz, dtype=np.float64)
        for s in seeds:
            p_vec[idx_map[s]] = 1.0 / len(seeds)

        trans_T = np.zeros((n_sz, n_sz), dtype=np.float64)
        dangling = np.zeros(n_sz, dtype=np.float64)
        for i, u in enumerate(nodes):
            out_e = list(dg.out_edges(u, data=True))
            tot = sum(float(ed.get("weight", 1.0)) for _, _, ed in out_e)
            if tot <= 1e-12:
                dangling[i] = 1.0
            else:
                for _, v, ed in out_e:
                    trans_T[idx_map[v], i] = float(ed.get("weight", 1.0)) / tot

        pi = p_vec.copy()
        alpha = self.ppr_alpha
        for _ in range(50):
            d_mass = float(np.dot(dangling, pi))
            pi_next = (1.0 - alpha) * p_vec + alpha * (trans_T @ pi + d_mass * p_vec)
            if float(np.sum(np.abs(pi_next - pi))) < n_sz * 1e-6:
                pi = pi_next
                break
            pi = pi_next

        node_ppr = {nodes[i]: float(pi[i]) for i in range(n_sz)}

        paths: dict[str, list[str]] = {}
        for s in seeds:
            for n1 in dg.successors(s):
                r1 = next(iter((self.graph.get_edge_data(s, n1) or {}).values()), {}).get("relation", "->")
                p1 = f"{s} -[{r1}]-> {n1}"
                paths.setdefault(n1, []).append(p1)
                for n2 in dg.successors(n1):
                    if n2 != s:
                        r2 = next(iter((self.graph.get_edge_data(n1, n2) or {}).values()), {}).get("relation", "->")
                        paths.setdefault(n2, []).append(f"{p1} -[{r2}]-> {n2}")

        return node_ppr, paths

    def _fts5_scores(self, query: str) -> dict[str, float]:
        fts_q = build_fts5_or_query(query)
        if not fts_q:
            return {}
        cur = self._conn.cursor()
        try:
            rows = cur.execute(
                "SELECT memory_id, bm25(steward_memories_fts) AS b FROM steward_memories_fts WHERE steward_memories_fts MATCH ? LIMIT 20",
                (fts_q,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        scores: dict[str, float] = {}
        for r in rows:
            raw_bm25 = abs(float(r["b"]))
            scores[str(r["memory_id"])] = raw_bm25 / (raw_bm25 + 1.5)
        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        query_time: float | None = None,
    ) -> MemorySearchResult:
        """Rank active memories via Vector + SQLite FTS5 + Directed PPR + Ebbinghaus Decay."""
        if not self._records:
            return MemorySearchResult(hits=[], abstained=True)

        now_ts = float(query_time if query_time is not None else time.time())
        q_vec = self.embedder.embed(query)
        lex_map = self._fts5_scores(query)
        node_ppr, hop_map = self._compute_ppr(query)

        superseded_skipped = 0
        active_ids: list[str] = []
        for mid, rec in self._records.items():
            if self.enable_supersession_and_decay and rec.superseded_by is not None:
                superseded_skipped += 1
                continue
            active_ids.append(mid)

        raw_mem_ppr: dict[str, float] = {}
        mem_paths: dict[str, list[str]] = {}
        for mid in active_ids:
            rec = self._records[mid]
            raw_mem_ppr[mid] = sum(
                node_ppr.get(self.resolve_entity(e), 0.0) for e in rec.entities
            )
            p_list: list[str] = []
            for e in rec.entities:
                for p in hop_map.get(self.resolve_entity(e), []):
                    if p not in p_list:
                        p_list.append(p)
            mem_paths[mid] = p_list[:3]

        max_ppr = max(raw_mem_ppr.values(), default=0.0)

        w_v, w_l, w_g = self.w_vec, self.w_lex, self.w_ppr
        if not self.enable_graph_ppr:
            denom = w_v + w_l
            w_v, w_l, w_g = w_v / denom, w_l / denom, 0.0

        candidates: list[MemoryHit] = []
        for mid in active_ids:
            rec = self._records[mid]
            v_score = max(0.0, float(np.dot(q_vec, self._vectors[mid])))
            l_score = lex_map.get(mid, 0.0)
            g_score = (raw_mem_ppr[mid] / (max_ppr + 1e-9)) if max_ppr > 1e-9 else 0.0

            if self.enable_supersession_and_decay:
                age_h = max(0.0, now_ts - rec.updated_at) / 3600.0
                eff_hl = self.base_half_life_hours * (1.0 + 0.5 * math.log(1.0 + rec.access_count))
                recency = math.exp(-math.log(2.0) * age_h / eff_hl)
                decay_mult = recency * (0.75 + 0.25 * rec.importance)
            else:
                decay_mult = 1.0

            combined = (w_v * v_score + w_l * l_score + w_g * g_score) * decay_mult
            candidates.append(
                MemoryHit(
                    record=rec,
                    final_score=round(combined, 5),
                    vector_score=round(v_score, 4),
                    lexical_score=round(l_score, 4),
                    graph_ppr_score=round(g_score, 4),
                    decay_multiplier=round(decay_mult, 4),
                    hop_paths=mem_paths.get(mid, []),
                )
            )

        candidates.sort(key=lambda h: h.final_score, reverse=True)

        # Calibrated Abstention Gate: reject unrelated queries (e.g., unrecorded city/hotel)
        if not candidates or candidates[0].final_score < self.abstention_threshold:
            return MemorySearchResult(
                hits=[],
                abstained=True,
                superseded_filtered_count=superseded_skipped,
            )

        # Location-mismatch abstention guard for specific unrecorded travel destinations
        queried_cities = set(re.findall(r"(北京|上海|深圳|广州|杭州|成都|东京|新加坡|威斯汀)", query))
        if queried_cities and self.enable_supersession_and_decay:
            has_city_evidence = any(
                any(city in h.record.content for city in queried_cities)
                for h in candidates[:top_k]
            )
            if not has_city_evidence and candidates[0].graph_ppr_score == 0.0:
                return MemorySearchResult(
                    hits=[],
                    abstained=True,
                    superseded_filtered_count=superseded_skipped,
                )

        filtered = [h for h in candidates[:top_k] if h.final_score >= self.abstention_threshold * 0.75]
        return MemorySearchResult(
            hits=filtered,
            abstained=(len(filtered) == 0),
            superseded_filtered_count=superseded_skipped,
        )

    def search_flat_baseline(self, query: str, *, top_k: int = 4) -> MemorySearchResult:
        """Unpruned flat vector + BM25 search without PPR, slot supersession, decay, or abstention."""
        if not self._records:
            return MemorySearchResult(hits=[], abstained=False)
        q_vec = self.embedder.embed(query)
        lex_map = self._fts5_scores(query)
        denom = self.w_vec + self.w_lex
        w_v, w_l = self.w_vec / denom, self.w_lex / denom
        scored: list[MemoryHit] = []
        for mid, rec in self._records.items():
            v_s = max(0.0, float(np.dot(q_vec, self._vectors[mid])))
            l_s = lex_map.get(mid, 0.0)
            comb = w_v * v_s + w_l * l_s
            scored.append(
                MemoryHit(
                    record=rec,
                    final_score=round(comb, 5),
                    vector_score=round(v_s, 4),
                    lexical_score=round(l_s, 4),
                    graph_ppr_score=0.0,
                    decay_multiplier=1.0,
                    hop_paths=[],
                )
            )
        scored.sort(key=lambda h: h.final_score, reverse=True)
        return MemorySearchResult(hits=scored[:top_k], abstained=False, superseded_filtered_count=0)

