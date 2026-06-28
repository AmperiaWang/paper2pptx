"""
论文演示文稿结构定义 — 加载 paper_structure.json 并注入 LLM 提示词。

为什么需要本模块？
  仅依赖 LLM「自由发挥」时，容易在优化语言风格后丢失章节完整性（例如只剩 3–4 页、
  跳过 Related Work / Conclusion）。paper_structure.json 是**固定的讲解骨架**，
  描述每个章节应回答哪些问题、建议页数；pipeline 将其格式化后写入 prompt，
  并在生成后做轻量审计，提醒缺失章节。

文件格式（paper_structure.json）：
  JSON 数组，每项包含：
    - title  : 章节英文名（Introduction / Related Work / …）
    - focus  : 该章节必须在幻灯片中覆盖的问题清单
    - pages  : 建议 content 页数量（如 "1-2 pages"）

与幻灯片 JSON 的对应关系：
  LLM 应输出 type=section（章节标题）+ 若干 type=content（正文页）。
  section 标题使用 format_section_label() 返回的中文/英文标签。
  flatten_sections() 随后会把 section 合并进 content，但**生成阶段必须保留完整 section**。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 默认结构文件：项目根目录 paper_structure.json
DEFAULT_STRUCTURE_PATH = Path(__file__).resolve().parent.parent / "paper_structure.json"

# paper_structure.json 中的英文章节名 -> 幻灯片 section 标题
SECTION_LABELS: dict[str, dict[str, str]] = {
    "zh_cn": {
        "Introduction": "引言",
        "Related Work": "相关工作",
        "Methods": "方法",
        "Experiment": "实验",
        "Conclusion": "结论",
    },
    "en_us": {
        "Introduction": "Introduction",
        "Related Work": "Related Work",
        "Methods": "Methods",
        "Experiment": "Experiments",
        "Conclusion": "Conclusion",
    },
}

# 用于 audit 时的别名（LLM 可能输出的变体）
SECTION_ALIASES: dict[str, frozenset[str]] = {
    "Introduction": frozenset({"introduction", "intro", "引言", "背景"}),
    "Related Work": frozenset({"related work", "related works", "相关工作", "背景工作"}),
    "Methods": frozenset({"methods", "method", "methodology", "方法", "模型"}),
    "Experiment": frozenset({"experiment", "experiments", "results", "evaluation", "实验", "结果"}),
    "Conclusion": frozenset({"conclusion", "conclusions", "summary", "结论", "总结"}),
}


def load_paper_structure(path: str | Path | None = None) -> list[dict]:
    """
    加载 paper_structure.json。

    参数:
        path: 结构文件路径；默认使用项目根目录下的 paper_structure.json。

    返回:
        章节字典列表，每项含 title / focus / pages。

    抛出:
        FileNotFoundError — 文件不存在。
        ValueError        — JSON 不是非空数组。
    """
    structure_path = Path(path) if path else DEFAULT_STRUCTURE_PATH
    if not structure_path.is_file():
        raise FileNotFoundError(f"Paper structure file not found: {structure_path}")

    with open(structure_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Paper structure must be a non-empty JSON array: {structure_path}")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "title" not in entry:
            raise ValueError(f"Invalid structure entry at index {i}: need 'title' field")

    return data


def format_section_label(section_title_en: str, lang: str = "zh_cn") -> str:
    """将 JSON 中的英文章节名转为幻灯片 section 标题。"""
    lang_key = "en_us" if lang.startswith("en") else "zh_cn"
    return SECTION_LABELS.get(lang_key, SECTION_LABELS["zh_cn"]).get(
        section_title_en, section_title_en
    )


def format_structure_for_prompt(sections: list[dict], lang: str = "zh_cn") -> str:
    """
    把结构 JSON 格式化为 LLM 可执行的「章节 + 必答问题 + 页数」清单。

    该字符串注入 analyze_paper / build_narrative 的 {paper_structure} 占位符。
    LLM 必须按顺序覆盖全部章节，且每个 focus 问题都要在对应章节的 bullets 中体现。
    """
    lang_key = "en_us" if lang.startswith("en") else "zh_cn"
    lines: list[str] = []

    if lang_key == "zh_cn":
        lines.append(
            "以下结构为**硬性要求**：必须按顺序包含全部 5 个 section，"
            "且每个 focus 问题都要在对应章节的 content 页 bullets 中回答（可合并到同页，不可整段跳过）。"
        )
    else:
        lines.append(
            "The outline below is **mandatory**: include all 5 sections in order, "
            "and address every focus question in that section's content bullets."
        )

    for idx, entry in enumerate(sections, start=1):
        en_title = entry["title"]
        label = format_section_label(en_title, lang)
        pages = entry.get("pages", "")
        focus_list = entry.get("focus") or []

        lines.append(f"\n### {idx}. Section「{label}」({en_title}) — {pages}")
        for q in focus_list:
            lines.append(f"- {q}")

    if lang_key == "zh_cn":
        lines.append(
            "\n生成 JSON 时：每个 section 先输出 "
            '`{"type":"section","title":"章节名"}`，再输出该章若干 content 页；'
            "Methods 按论文核心创新点拆分（每个创新点 1–2 页）；"
            "Experiment 按实验/数据集拆分（共 2–3 页），含 table/figure 引用。"
        )
    else:
        lines.append(
            "\nIn JSON: emit a section marker then multiple content slides per block; "
            "split Methods by key insight; Experiments 2–3 pages with table/figure refs."
        )

    return "\n".join(lines)


def _normalize_section_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _matches_section(slide_title: str, canonical_en: str) -> bool:
    """判断幻灯片 section/content 标题是否属于某 canonical 章节。"""
    norm = _normalize_section_name(slide_title)
    aliases = SECTION_ALIASES.get(canonical_en, frozenset())
    return any(alias in norm or norm in alias for alias in aliases)


def audit_slide_structure(
    slides: list[dict],
    sections: list[dict],
    lang: str = "zh_cn",
) -> list[str]:
    """
    检查 LLM 输出的 slides 是否覆盖 paper_structure 中的全部章节。

    在 flatten_sections 之前调用：统计 type=section 的标题，
    以及 content 页上附带的 section 字段（若 LLM 已填写）。

    返回:
        警告信息列表（空列表表示结构完整）。
    """
    found: set[str] = set()

    for slide in slides:
        if slide.get("type") == "section":
            title = slide.get("title") or ""
            for entry in sections:
                if _matches_section(title, entry["title"]) or _matches_section(
                    title, format_section_label(entry["title"], lang)
                ):
                    found.add(entry["title"])

        if slide.get("type") == "content":
            title = slide.get("section") or slide.get("title") or ""
            for entry in sections:
                if _matches_section(title, entry["title"]) or _matches_section(
                    title, format_section_label(entry["title"], lang)
                ):
                    found.add(entry["title"])

    warnings: list[str] = []
    for entry in sections:
        if entry["title"] not in found:
            label = format_section_label(entry["title"], lang)
            warnings.append(f"Missing section: {label} ({entry['title']})")

    return warnings
