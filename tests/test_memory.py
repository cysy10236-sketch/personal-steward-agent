"""Unit tests for StewardMemoryEngine: SQLite WAL, Supersession, NumPy PPR, and Abstention."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.eval.run_benchmark import seed_steward_memories
from src.memory.steward_memory import EntityTriple, StewardMemoryEngine


def test_idempotency_and_sqlite_reload(tmp_path: Path) -> None:
    db_file = tmp_path / "persist.sqlite3"
    eng1 = StewardMemoryEngine(db_file)
    r1 = eng1.add_memory_sync("深圳出差高铁 G79 二等座票价 936 元。", timestamp=1763400000.0)
    r2 = eng1.add_memory_sync("  深圳出差高铁 G79 二等座票价 936 元。 ", timestamp=1763400100.0)
    assert r1.memory_id == r2.memory_id
    assert r2.access_count == 2
    eng1._conn.close()

    eng2 = StewardMemoryEngine(db_file)
    res = eng2.search("深圳出差高铁 G79 票价", top_k=2, query_time=1763400200.0)
    assert not res.abstained
    assert res.hits[0].record.memory_id == r1.memory_id
    assert "936" in res.hits[0].record.content
    eng2._conn.close()


def test_slot_supersession_and_directed_ppr_over_distractors(tmp_path: Path) -> None:
    eng = StewardMemoryEngine(tmp_path / "ppr.sqlite3")
    seed_steward_memories(eng, anchor_ts=1763400000.0)

    # 1. Flight CA1301 must be superseded by Rail G79
    r_leg = eng.search("11月18日去深圳出差最新车次与票价是多少？", top_k=3, query_time=1763400500.0)
    got_leg_ids = [h.record.memory_id for h in r_leg.hits]
    assert "mem_flight_new" in got_leg_ids
    assert "mem_flight_old" not in got_leg_ids
    assert r_leg.superseded_filtered_count >= 2

    # 2. 2-hop Directed PPR must surface mem_pangu_rule (850 CNY & 陈总监) despite 5 homogeneous distractors
    r_ppr = eng.search(
        "我这趟「华南智算中心拜访」的一线城市住宿差标上限是多少元？报销终审审批人是谁？",
        top_k=4,
        query_time=1763400500.0,
    )
    got_ppr_ids = [h.record.memory_id for h in r_ppr.hits]
    assert "mem_visit_bridge" in got_ppr_ids
    assert "mem_pangu_rule" in got_ppr_ids

    # Probability mass conservation check on dangling graph
    node_ppr, _ = eng._compute_ppr("华南智算中心拜访")
    assert abs(sum(node_ppr.values()) - 1.0) < 1e-4

    # 3. Unrecorded destination abstention
    r_abs = eng.search("上个月去成都威斯汀酒店住宿花了多少元？", top_k=4, query_time=1763400500.0)
    assert r_abs.abstained is True
    assert r_abs.hits == []
    eng._conn.close()


@pytest.mark.asyncio
async def test_async_read_your_writes(tmp_path: Path) -> None:
    eng = StewardMemoryEngine(tmp_path / "async_mem.sqlite3")
    rec = await eng.add_memory_async(
        "「极光多模态项目」差旅特批审批人为韩总监。",
        entities=["极光多模态项目", "韩总监"],
        relations=[EntityTriple(source="极光多模态项目", relation="approved_by", target="韩总监")],
    )
    res = eng.search("极光多模态项目的审批人是谁？", top_k=2)
    assert not res.abstained
    assert res.hits[0].record.memory_id == rec.memory_id
    await eng.flush_and_close()
