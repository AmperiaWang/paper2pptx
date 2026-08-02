"""
Table 与幻灯片的自动匹配。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paper2ppt.table_extract import PaperTable

TABLE_REF_RE = re.compile(r"(?:table|表)\s*(\d+)", re.IGNORECASE)
TABLE_ID_RE = re.compile(r"table\s*(\d+)", re.IGNORECASE)

TABLE_SECTIONS = frozenset({
    "experiments", "experiment", "experimental", "results", "evaluation",
    "实验", "结果",
})


def normalize_table_id(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    m = TABLE_ID_RE.search(value)
    if m:
        return f"table{int(m.group(1))}"
    if value.lower().startswith("table") and value[5:].isdigit():
        return f"table{int(value[5:])}"
    return None


def slide_text(slide: dict) -> str:
    parts = [slide.get("title") or "", slide.get("bridge") or ""]
    parts.extend(slide.get("bullets") or [])
    return " ".join(parts)


def find_table_refs(text: str) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for m in TABLE_REF_RE.finditer(text):
        num = int(m.group(1))
        if num not in seen:
            seen.add(num)
            result.append(num)
    return result


def _section_allows(section_title: str) -> bool:
    lower = section_title.lower()
    return any(key in lower for key in TABLE_SECTIONS)


def assign_tables_to_slides(
    slides: list[dict],
    tables: list[PaperTable],
) -> int:
    """为实验相关幻灯片分配 table 字段。"""
    if not tables:
        for slide in slides:
            slide.pop("table", None)
        return 0

    available = {t.table_id: t for t in tables}
    used: set[str] = set()
    current_section = ""
    content_items: list[tuple[dict, str]] = []

    for slide in slides:
        if slide.get("type") == "section":
            current_section = slide.get("title") or ""
        elif slide.get("type") == "content":
            content_items.append((slide, slide.get("section") or current_section))

    for slide in slides:
        if slide.get("type") != "content":
            slide.pop("table", None)

    for slide, section in content_items:
        tid = normalize_table_id(slide.get("table"))
        if tid and tid in available and tid not in used:
            slide["table"] = tid
            used.add(tid)
        else:
            slide.pop("table", None)

    for slide, section in content_items:
        if slide.get("table"):
            continue
        if slide.get("figure"):
            continue
        if not _section_allows(section):
            continue
        for num in find_table_refs(slide_text(slide)):
            tid = f"table{num}"
            if tid in available and tid not in used:
                slide["table"] = tid
                used.add(tid)
                break

    for table in sorted(tables, key=lambda t: t.number):
        if table.table_id in used:
            continue
        for slide, section in content_items:
            if slide.get("table"):
                continue
            if slide.get("figure"):
                continue
            if not _section_allows(section):
                continue
            caption_words = set(re.findall(r"[a-zA-Z]{4,}", table.caption.lower()))
            text_lower = slide_text(slide).lower()
            if caption_words and sum(1 for w in caption_words if w in text_lower) >= 2:
                slide["table"] = table.table_id
                used.add(table.table_id)
                break

    # 兜底：未分配的 Table 依次放到实验章节的无表 content 页
    eligible = [
        slide
        for slide, section in content_items
        if not slide.get("table") and not slide.get("figure") and _section_allows(section)
    ]
    for table in sorted(tables, key=lambda t: t.number):
        if table.table_id in used or not eligible:
            continue
        target = eligible.pop(0)
        target["table"] = table.table_id
        used.add(table.table_id)

    return sum(1 for slide, _ in content_items if slide.get("table"))
