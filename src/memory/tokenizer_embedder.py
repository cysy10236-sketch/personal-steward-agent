"""Bilingual embedder, SQLite FTS5 segmenter, general anchor & travel slot extractor."""

from __future__ import annotations

import hashlib
import math
import re
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{1,}|[\u4e00-\u9fff]+", re.I)
_EN_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{1,}\b|\b[A-Za-z]+-[A-Za-z0-9_-]+\b")
_QUOTED_RE = re.compile(r"[「『“\"']([^」』”\"'\n]{2,24})[」』”\"']")
_CJK_ENTITY_RE = re.compile(
    r"([\u4e00-\u9fff]{2,7}(?:项目|二期|一期|三期|中心|智算|集团|酒店|宾馆|部门|峰会|拜访|实验室|复盘会|评审会|周会|对齐会))|"
    r"((?:张|王|李|赵|陈|刘|杨|黄|周|吴|林|徐|孙|马|胡|郭|何|高|罗|郑|谢|沈|韩)[\u4e00-\u9fff]{1,2})"
)
_ANCHOR_VENUE_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z]{2,10}(?:酒店|宾馆|饭店|机场|项目|中心|实验室|峰会|拜访|复盘会|评审会|周会|对齐会))"
)
_ANCHOR_DEST_RE = re.compile(r"(?:去|赴|飞|在)([\u4e00-\u9fff]{2,4})(?:出差|住宿|入住|开会|拜访|酒店)")

_DOMAIN_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("航班", "高铁", "车次", "机票", "改签", "出发", "行程", "flight", "train", "itinerary", "depart"),
    ("酒店", "住宿", "房费", "水单", "入住", "退房", "差标", "hotel", "lodging", "folio", "cap"),
    ("发票", "报销", "税额", "水单", "打车", "网约车", "餐饮", "receipt", "invoice", "reimburse", "claim", "taxi"),
    ("预算", "经费", "额度", "超标", "打款", "转账", "汇率", "budget", "expense", "limit", "wire", "currency"),
    ("审批", "负责人", "主管", "特批", "归属", "关联", "approver", "owner", "lead", "project"),
    ("会议", "日程", "排期", "复盘", "拜访", "改期", "meeting", "calendar", "schedule", "visit"),
    ("身份证", "银行卡", "卡号", "手机", "隐私", "实名", "id_card", "bank", "privacy", "pii"),
)

_STOPWORDS = frozenset(
    {"我们", "请问", "帮我", "查一下", "多少", "什么", "最新", "目前", "上个月", "这趟", "这张", "可以", "一般", "所有", "标准"}
)
_GENERIC_ANCHORS = frozenset({"酒店", "项目", "中心", "城市酒店", "一线城市", "出差酒店", "住宿酒店", "差旅项目", "报销项目"})


def estimate_tokens(text: str) -> int:
    if not text:
        return 1
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return max(1, int(math.ceil(cjk * 0.85 + (len(text) - cjk) / 3.8)))


def fts5_segment_text(text: str) -> str:
    """Segment CJK into bigrams alongside Latin tokens for SQLite FTS5 BM25."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if "\u4e00" <= tok[0] <= "\u9fff":
            if len(tok) == 1:
                out.append(tok)
            else:
                for i in range(len(tok) - 1):
                    bg = tok[i : i + 2]
                    if bg not in _STOPWORDS:
                        out.append(bg)
        else:
            out.append(tok)
    return " ".join(out)


def build_fts5_or_query(query: str) -> str:
    cleaned = [re.sub(r"[^a-z0-9\u4e00-\u9fff_-]", "", t) for t in fts5_segment_text(query).split()]
    uniq = list(dict.fromkeys(t for t in cleaned if len(t) >= 2))[:14]
    return " OR ".join(f'"{t}"' for t in uniq) if uniq else ""


def extract_query_anchors(query: str) -> list[str]:
    """Extract domain-agnostic specific entity anchors from a query (zero hardcoded brand lists)."""
    anchors: list[str] = []
    for m in _QUOTED_RE.finditer(query):
        val = m.group(1).strip()
        if len(val) >= 2 and val not in _GENERIC_ANCHORS and val not in anchors:
            anchors.append(val)
    for m in _ANCHOR_VENUE_RE.finditer(query):
        val = re.sub(r"^(?:的|在|去|赴|飞|这趟|上次|上个月|那次|对应)+", "", m.group(1).strip())
        if len(val) >= 3 and val not in _GENERIC_ANCHORS and val not in anchors:
            anchors.append(val)
    for m in _ANCHOR_DEST_RE.finditer(query):
        val = re.sub(r"^(?:去|赴|飞|在|上次|上个月)+", "", m.group(1).strip())
        if len(val) >= 2 and val not in _STOPWORDS and val not in anchors:
            anchors.append(val)
    for m in re.finditer(r"\b(?:[A-Z]{1,2}\d{2,4}|PRJ-[A-Z0-9-]+|INV-\d+)\b", query):
        if m.group(0) not in anchors:
            anchors.append(m.group(0))
    return anchors


def infer_travel_slot(text: str) -> tuple[str | None, str | None]:
    """Detect general single-valued mutable personal slots without hardcoding city names."""
    # 1. Trip transport leg per destination (e.g. "深圳出差...航班/高铁")
    dest_m = re.search(r"([\u4e00-\u9fff]{2,3})出差", text) or re.search(
        r"(?:赴|飞|去)([\u4e00-\u9fff]{2,3})(?:的|出差|行程|航班|高铁)", text
    )
    if dest_m and any(k in text for k in ("航班", "高铁", "车次", "改签", "退掉")):
        dest = re.sub(r"^(?:我|去|赴|飞|在)+", "", dest_m.group(1))
        if len(dest) >= 2:
            return f"trip_{dest}", "active_transport_leg"

    # 2. Trip booked hotel per destination
    if dest_m and any(k in text for k in ("改订", "预订酒店", "入住酒店", "更换酒店")):
        dest = re.sub(r"^(?:我|去|赴|飞|在)+", "", dest_m.group(1))
        if len(dest) >= 2:
            return f"trip_{dest}", "booked_hotel"

    # 3. Tiered lodging policy cap revision
    tier_m = re.search(r"(一线城市|二线城市|海外)(?:住宿)?差标", text)
    if tier_m and any(k in text for k in ("上限", "上调", "调整", "标准", "改为")):
        return f"policy_{tier_m.group(1)}", "hotel_nightly_cap"

    # 4. Named meeting reschedule slot
    meet_m = re.search(r"([\u4e00-\u9fffA-Za-z0-9_-]{2,12}(?:复盘会|评审会|周会|对齐会))", text)
    if meet_m and any(k in text for k in ("定在", "改期", "调整至", "排期", "周")):
        return meet_m.group(1), "meeting_slot"

    # 5. Project approver handover
    proj_m = re.search(r"[「『]?([\u4e00-\u9fffA-Za-z0-9_-]{2,10}(?:项目|二期|一期))[」』]?", text)
    if proj_m and any(k in text for k in ("审批人变更为", "审批人交接", "新任审批人", "原审批人")):
        return proj_m.group(1), "project_approver"

    # 6. Base office city relocation
    if any(k in text for k in ("常驻办公地", "现居城市", "搬迁至", "常驻地变更为")):
        return "user_profile", "base_office_city"

    return None, None


def extract_bilingual_entities(
    text: str,
    alias_register_cb: callable | None = None,
) -> list[str]:
    """Extract CJK & English travel/project/person entities and parenthetical aliases."""
    entities: list[str] = []
    for m in re.finditer(
        r"([\u4e00-\u9fffA-Za-z0-9_-]{2,16})\s*[（(]\s*([A-Za-z0-9_\u4e00-\u9fff -]{2,26})\s*[）)]",
        text,
    ):
        left = re.sub(r"^[的之是在于和与从去赴]+", "", m.group(1).strip())
        right = m.group(2).strip()
        if left not in _STOPWORDS and right not in _STOPWORDS:
            if alias_register_cb is not None:
                alias_register_cb(left, right)
            if left not in entities:
                entities.append(left)

    for m in _QUOTED_RE.finditer(text):
        val = m.group(1).strip()
        if val and val not in _STOPWORDS and val not in entities:
            entities.append(val)

    for m in _EN_ENTITY_RE.finditer(text):
        val = m.group(0).strip()
        if len(val) >= 2 and val not in entities:
            entities.append(val)

    for m in _CJK_ENTITY_RE.finditer(text):
        raw = re.sub(r"^[的之是在于和与从去赴]+", "", (m.group(1) or m.group(2) or "").strip())
        if len(raw) >= 2 and raw not in _STOPWORDS and raw not in entities:
            entities.append(raw)
    return entities


class DeterministicBilingualEmbedder:
    """Zero-dependency L2-normalized bilingual concept + N-gram hash embedder."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 64:
            raise ValueError("Embedding dim must be >= 64")
        self.dim = dim
        self._concept_slots = min(24, dim // 4)
        self._hash_slots = dim - self._concept_slots

    def _hash_idx(self, feat: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(digest, "little", signed=False)
        return v % self._hash_slots, (1.0 if ((v >> 63) & 1) == 0 else -1.0)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        s = (text or "").strip().lower()
        if not s:
            vec[0] = 1.0
            return vec

        for c_i, words in enumerate(_DOMAIN_CLUSTERS):
            hits = sum(1 for w in words if w in s)
            if hits > 0:
                vec[c_i % self._concept_slots] += float(min(2.8, 1.0 + 0.45 * hits))

        tokens = _TOKEN_RE.findall(s)
        feats: list[tuple[str, float]] = []
        for tok in tokens:
            if "\u4e00" <= tok[0] <= "\u9fff":
                for ch in tok:
                    feats.append((f"c1:{ch}", 0.35))
                for i in range(len(tok) - 1):
                    feats.append((f"c2:{tok[i:i+2]}", 1.40))
                for i in range(len(tok) - 2):
                    feats.append((f"c3:{tok[i:i+3]}", 1.65))
            else:
                feats.append((f"w:{tok}", 1.45))

        for i in range(len(tokens) - 1):
            feats.append((f"bg:{tokens[i]}_{tokens[i+1]}", 0.85))

        offset = self._concept_slots
        for feat, w in feats:
            idx, sign = self._hash_idx(feat)
            vec[offset + idx] += sign * w

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec
