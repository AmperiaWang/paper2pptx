"""
Figure 与幻灯片的自动匹配。

小模型（如 qwen3:4b）常不在 JSON 中填写 \"figure\" 字段，因此需要在 LLM
输出之后，根据幻灯片文字与 Figure 图注做程序化匹配。

规则：
  1. 保留并规范化 LLM 已填写的 figure id
  2. 在标题/bullets 中检测显式引用（Figure 3 / Fig. 3 / 图3）
  3. 用图注关键词与幻灯片文本的重叠度为未分配的 Figure 找最佳页面
  4. 无合适匹配时不插入图片（Introduction / Related Works 等纯文字页）
  5. 每张 Figure 最多分配一页，避免重复
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paper2ppt.pdf_parser import PaperFigure

# 幻灯片正文中对 Figure 的显式引用
FIGURE_REF_RE = re.compile(
    r"(?:figure|fig\.?|图)\s*(\d+)",
    re.IGNORECASE,
)

# 匹配 figure id 字符串：figure1 / Figure 1 / FIGURE3
FIGURE_ID_RE = re.compile(r"figure\s*(\d+)", re.IGNORECASE)

# 允许通过图注关键词 / LLM 自动匹配 Figure 的章节
FIGURE_AUTO_ASSIGN_SECTIONS = frozenset({
    "methods", "method", "methodology",
    "experiments", "experiment", "experimental", "results", "evaluation",
    "方法", "实验", "结果",
})

# 关键词匹配最低分数（Methods/Experiments 章节内）
MIN_CAPTION_MATCH_SCORE = 1

# 图注/幻灯片中无意义的停用词
STOPWORDS = frozenset(
    """
    a an the and or of in on at to for with by from as is are was were be been
    being this that these those our we they it its their an
    figure fig table section page model network task tasks
    的 了 在 是 与 及 等 对 中 为 从 将 被 所 一个 这种 通过 使用 进行 我们 本文 该 其
    """.split()
)


def normalize_figure_id(value: str | None) -> str | None:
    """将 LLM 可能返回的各种 figure 写法统一为 figureN。"""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    m = FIGURE_ID_RE.search(value)
    if m:
        return f"figure{int(m.group(1))}"
    if value.lower().startswith("figure") and value[6:].isdigit():
        return f"figure{int(value[6:])}"
    return None


def slide_text(slide: dict) -> str:
    """拼接幻灯片标题与 bullets 为一段 searchable 文本。"""
    parts = [slide.get("title") or ""]
    parts.extend(slide.get("bullets") or [])
    return " ".join(parts)


def find_figure_refs(text: str) -> list[int]:
    """从文本中提取所有 Figure 序号（去重保序）。"""
    seen: set[int] = set()
    result: list[int] = []
    for m in FIGURE_REF_RE.finditer(text):
        num = int(m.group(1))
        if num not in seen:
            seen.add(num)
            result.append(num)
    return result


def _caption_keywords(caption: str) -> set[str]:
    """从图注提取可用于匹配的关键词。"""
    words = re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", caption.lower())
    return {w for w in words if w not in STOPWORDS}


def caption_match_score(text: str, figure: PaperFigure) -> int:
    """计算幻灯片文本与 Figure 图注的关键词重叠数。"""
    text_lower = text.lower()
    keywords = _caption_keywords(figure.caption)
    if not keywords:
        return 0
    score = sum(1 for kw in keywords if kw in text_lower)
    return score


def _section_allows_auto_assign(section_title: str) -> bool:
    """Introduction / Related Works 等章节不允许自动插图，除非正文显式引用 Figure。"""
    lower = section_title.lower()
    return any(key in lower for key in FIGURE_AUTO_ASSIGN_SECTIONS)


def _is_experiment_section(section_title: str) -> bool:
    lower = section_title.lower()
    return any(key in lower for key in ("experiment", "evaluation", "results", "实验", "结果"))


def _iter_content_with_section(slides: list[dict]):
    """按顺序遍历 content 页，并携带所属章节标题。"""
    current_section = ""
    for slide in slides:
        if slide.get("type") == "section":
            current_section = slide.get("title") or ""
        elif slide.get("type") == "content":
            section = slide.get("section") or current_section
            yield slide, section


def assign_figures_to_slides(
    slides: list[dict],
    figures: list[PaperFigure],
    *,
    table_reserve: int = 0,
) -> int:
    """
    为 content 类型幻灯片分配 figure 字段。

    返回:
        成功分配 Figure 的幻灯片数量。
    """
    if not figures:
        for slide in slides:
            slide.pop("figure", None)
        return 0

    available = {f.figure_id: f for f in figures}
    used: set[str] = set()
    content_with_section = list(_iter_content_with_section(slides))
    experiment_slides = [
        slide for slide, section in content_with_section if _is_experiment_section(section)
    ]
    reserved_experiment_pages = min(table_reserve, len(experiment_slides) // 2)
    max_experiment_figures = len(experiment_slides) - reserved_experiment_pages

    def experiment_has_room() -> bool:
        assigned = sum(bool(slide.get("figure")) for slide in experiment_slides)
        return assigned < max_experiment_figures

    # 清除非 content 页及历史 figure 字段，再重新分配
    for slide in slides:
        if slide.get("type") != "content":
            slide.pop("figure", None)

    # Pass 1：保留 LLM 已填写的有效 figure（不限章节）
    for slide, _section in content_with_section:
        if slide.get("table") or slide.get("table_data"):
            slide.pop("figure", None)
            continue
        fid = normalize_figure_id(slide.get("figure"))
        if fid and fid in available and fid not in used:
            slide["figure"] = fid
            used.add(fid)
        else:
            slide.pop("figure", None)

    # Pass 2：正文显式 Figure 引用（任意章节均有效）
    for slide, _section in content_with_section:
        if slide.get("figure"):
            continue
        for num in find_figure_refs(slide_text(slide)):
            fid = f"figure{num}"
            if fid in available and fid not in used:
                slide["figure"] = fid
                used.add(fid)
                break

    # Pass 3：图注关键词匹配（仅 Methods/Experiments 章节）
    for figure in sorted(figures, key=lambda f: f.number):
        if figure.figure_id in used:
            continue
        best_slide: dict | None = None
        best_score = 0
        for slide, section in content_with_section:
            if slide.get("figure") or not _section_allows_auto_assign(section):
                continue
            if slide.get("table") or slide.get("table_data"):
                continue
            if _is_experiment_section(section) and not experiment_has_room():
                continue
            score = caption_match_score(slide_text(slide), figure)
            if score > best_score:
                best_score = score
                best_slide = slide
        if best_slide and best_score >= MIN_CAPTION_MATCH_SCORE:
            best_slide["figure"] = figure.figure_id
            used.add(figure.figure_id)

    # Pass 4：兜底 — 将未分配的 Figure 依次放到方法/实验章节的无图 content 页
    eligible = [
        slide
        for slide, section in content_with_section
        if not slide.get("figure")
        and not slide.get("table")
        and not slide.get("table_data")
        and _section_allows_auto_assign(section)
    ]
    for figure in sorted(figures, key=lambda f: f.number):
        if figure.figure_id in used or not eligible:
            continue
        target = next(
            (
                slide
                for slide in eligible
                if not _is_experiment_section(slide.get("section") or "")
                or experiment_has_room()
            ),
            None,
        )
        if target is None:
            break
        eligible.remove(target)
        target["figure"] = figure.figure_id
        used.add(figure.figure_id)

    content_slides = [s for s, _ in content_with_section]
    return sum(1 for s in content_slides if s.get("figure"))
