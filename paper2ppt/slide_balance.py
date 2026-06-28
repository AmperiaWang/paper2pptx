"""
幻灯片篇幅平衡 — 防止文字溢出，必要时拆分为多页。

采用简单启发式：
  - 限制标题/单条 bullet 最大字符数
  - 限制每页 bullet 条数与总字符数（有图页更紧）
  - 超限时拆成「续」页，续页不重复插图
  - 生成阶段再根据字数动态缩小字号
"""

from __future__ import annotations

import copy
import re

# 单页容量（与 prompt 中 3–5 条分析性 bullet 对齐）
LIMITS = {
    "no_figure": {"max_bullets": 5, "max_chars": 450, "max_bullet_chars": 100},
    "with_figure": {"max_bullets": 4, "max_chars": 320, "max_bullet_chars": 90},
}
MAX_TITLE_CHARS = 36


def balance_slides(slides: list[dict], lang: str = "zh_cn") -> list[dict]:
    """
    调整 slides 列表：截断过长文字，并将过载 content 页拆成多页。

    返回新的 slides 列表（不修改原列表）。
    """
    result: list[dict] = []
    cont_suffix = "（续）" if lang.startswith("zh") else " (cont.)"

    for slide in slides:
        slide_type = slide.get("type", "content")
        if slide_type != "content":
            result.append(copy.deepcopy(slide))
            continue

        base = copy.deepcopy(slide)
        base["title"] = _truncate(base.get("title", ""), MAX_TITLE_CHARS)
        has_figure = bool(base.get("figure"))
        limits = LIMITS["with_figure" if has_figure else "no_figure"]

        bullets = [
            _truncate(b, limits["max_bullet_chars"])
            for b in base.get("bullets", [])
            if str(b).strip()
        ]

        chunks = _split_into_pages(
            bullets,
            max_bullets=limits["max_bullets"],
            max_chars=limits["max_chars"],
        )

        for idx, chunk in enumerate(chunks):
            page = copy.deepcopy(base)
            if idx > 0:
                page["title"] = _truncate(
                    f"{base['title']}{cont_suffix}", MAX_TITLE_CHARS + 4
                )
                page.pop("figure", None)
            page["bullets"] = chunk
            result.append(page)

    return result


def estimate_content_font_size(bullets: list[str], has_figure: bool) -> int:
    """根据 bullet 数量与总字数估算正文字号（Pt）。"""
    count = len(bullets)
    total = sum(len(b) for b in bullets)
    if has_figure:
        if total > 240 or count > 4:
            return 13
        if total > 180 or count > 3:
            return 14
        return 15
    if total > 380 or count > 6:
        return 13
    if total > 280 or count > 5:
        return 14
    if total > 200:
        return 16
    return 18


def estimate_title_font_size(title: str, default: int = 28) -> int:
    """标题过长时缩小字号。"""
    n = len(title)
    if n > 40:
        return max(default - 8, 20)
    if n > 28:
        return default - 4
    return default


def _split_into_pages(
    bullets: list[str],
    *,
    max_bullets: int,
    max_chars: int,
) -> list[list[str]]:
    """将 bullets 拆成多页，每页不超过条数与总字符上限。"""
    if not bullets:
        return [[]]

    pages: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for bullet in bullets:
        b_len = len(bullet)
        would_exceed_count = len(current) >= max_bullets
        would_exceed_chars = current and current_chars + b_len > max_chars

        if current and (would_exceed_count or would_exceed_chars):
            pages.append(current)
            current = []
            current_chars = 0

        current.append(bullet)
        current_chars += b_len

    if current:
        pages.append(current)

    return pages


def _truncate(text: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
