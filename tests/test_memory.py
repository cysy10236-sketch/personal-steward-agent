"""Unit tests for StewardMemoryEngine: SQLite WAL, Supersession, NumPy PPR & generic abstention."""

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
    assert r1.memory_id == r2.memory_id and r2.access_count == 2
    eng1._conn.close()

    eng2 = StewardMemoryEngine(db_file)
    res = eng2.search("深圳出差高铁 G79 票价", top_k=2, query_time=1763400200.0)
    assert not res.abstained and res.hits[0].record.memory_id == r1.memory_id
    eng2._conn.close()


def test_slot_supersession_ppr_and_generic_abstention(tmp_path: Path) -> None:
    eng = StewardMemoryEngine(tmp_path / "ppr.sqlite3")
    seed_steward_memories(eng, anchor_ts=1763400000.0)

    # 1. Supersession: flight CA1301 -> rail G79
    r_leg = eng.search("11月18日去深圳出差最新车次与票价是多少？", top_k=3, query_time=1763400500.0)
    got_leg = [h.record.memory_id for h in r_leg.hits]
    assert "mem_flight_new" in got_leg and "mem_flight_old" not in got_leg

    # 2. 2-Hop Directed PPR over 8 homogeneous distractors (凌云视觉拜访 -> 凌云视觉项目 -> 920元 & 沈总监)
    r_ppr = eng.search("我这趟「凌云视觉拜访」的一线城市住宿差标上限是多少元？审批人是谁？", top_k=4, query_time=1763400500.0)
    got_ppr = [h.record.memory_id for h in r_ppr.hits]
    assert "mem_ly_bridge" in got_ppr and "mem_ly_rule" in got_ppr

    node_ppr, _ = eng._compute_ppr("凌云视觉拜访")
    assert abs(sum(node_ppr.values()) - 1.0) < 1e-4

    # 3. Generic Entity-Coverage Abstention on unseen city/hotel (zero hardcoded city names!)
    for unseen_q in (
        "我上个月去哈尔滨索菲特酒店出差住宿花了多少元？",
        "「量子雷达项目」的审批人是谁？",
        "我在南京金陵饭店的发票金额是多少？",
    ):
        r_abs = eng.search(unseen_q, top_k=4, query_time=1763400500.0)
        assert r_abs.abstained is True and r_abs.hits == []
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
    assert not res.abstained and res.hits[0].record.memory_id == rec.memory_id
    await eng.flush_and_close()
