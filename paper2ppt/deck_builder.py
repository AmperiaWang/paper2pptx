"""
多阶段幻灯片构建 — 分片读透论文，再按 paper_structure 逐章生成高质量组会 PPT。

流水线（相比旧版单次 analyze_paper 的改进）：
  1. deep_read_paper   — 分片精读，产出带数字/图表信息的结构化笔记
  2. extract_metadata  — 单独提取标题、作者（保证 title 页）
  3. generate_section  — **每章一次 LLM**，只写该章 slides，降低幻觉与结构丢失
  4. assemble + polish — 合并、补 title 页、补缺失页标题
  5. figure/table 匹配 — 由 pipeline 在 flatten 后执行

设计原则：
  - 事实仅来自论文笔记，章节 prompt 中反复强调禁止编造
  - 每章独立生成，上下文更小，模型更不易「跑题」
  - 程序侧保证 title 页、页标题、Figure 兜底分配
"""

from __future__ import annotations

import re

from tqdm import tqdm

from paper2ppt.llm.base import LLMBackend
from paper2ppt.llm.ollama import OllamaBackend
from paper2ppt.paper_chunk import (
    CHUNK_NUM_CTX,
    CHUNK_TIMEOUT,
    SYNTHESIS_NUM_CTX,
    SYNTHESIS_TIMEOUT,
    merge_chunk_notes,
    split_text_chunks,
    summarize_chunks,
)
from paper2ppt.paper_structure import (
    format_section_label,
    format_structure_for_prompt,
)
from paper2ppt.pdf_parser import PaperContent, build_figure_catalog
from paper2ppt.prompts import extract_json
from paper2ppt.slide_polish import polish_slides
from paper2ppt.table_extract import build_table_catalog


def _backend_chat(
    backend: LLMBackend,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    num_ctx: int | None = None,
    timeout: int = 600,
) -> str:
    if isinstance(backend, OllamaBackend):
        return backend.chat(
            messages,
            temperature=temperature,
            num_ctx=num_ctx,
            timeout=timeout,
        )
    return backend.chat(messages, temperature=temperature)


def deep_read_paper(
    backend: LLMBackend,
    paper_text: str,
    prompts: dict[str, str],
) -> str:
    """
    分片精读全文。短论文也走至少 1 次 summarize，长论文按 chunk 多次阅读。
    """
    if len(paper_text.strip()) < 100:
        return paper_text
    chunks = split_text_chunks(paper_text)
    if len(chunks) == 1:
        # 单段仍用 summarize_chunk 模板做「读透」而非原文直塞
        template = prompts.get("summarize_chunk", "")
        messages = [
            {"role": "system", "content": prompts["system"]},
            {
                "role": "user",
                "content": template.format(index=1, total=1, chunk_text=chunks[0]),
            },
        ]
        note = _backend_chat(
            backend,
            messages,
            temperature=0.2,
            num_ctx=CHUNK_NUM_CTX,
            timeout=CHUNK_TIMEOUT,
        )
        merged = merge_chunk_notes([note.strip()])
        print(f"      Deep-read notes ready ({len(merged)} chars)")
        return merged
    return summarize_chunks(backend, paper_text, prompts)


def extract_metadata(
    backend: LLMBackend,
    paper: PaperContent,
    paper_notes: str,
    prompts: dict[str, str],
) -> dict[str, str]:
    """从笔记提取 title / authors；失败时回退 PDF 启发式标题。"""
    template = prompts.get("extract_metadata")
    title = (paper.title or "").strip()
    authors = ""

    if template:
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": template.format(paper_notes=paper_notes[:12000])},
        ]
        try:
            data = extract_json(
                _backend_chat(
                    backend,
                    messages,
                    temperature=0.1,
                    num_ctx=CHUNK_NUM_CTX,
                    timeout=CHUNK_TIMEOUT,
                )
            )
            title = (data.get("title") or title or "Research Paper").strip()
            authors = (data.get("authors") or "").strip()
        except Exception:
            pass

    if not title:
        title = "Research Paper"
    return {"title": title, "authors": authors}


def _min_pages(pages_hint: str) -> int:
    """从 '1-2 pages' / '2-3 pages' 解析最少 content 页数。"""
    nums = [int(x) for x in re.findall(r"\d+", pages_hint or "")]
    if not nums:
        return 1
    return max(nums[0], nums[-1] if len(nums) > 1 else nums[0])


def _format_focus_list(focus: list[str]) -> str:
    return "\n".join(f"- {q}" for q in focus)


def generate_section_slides(
    backend: LLMBackend,
    section_entry: dict,
    *,
    paper_notes: str,
    figure_catalog: str,
    table_catalog: str,
    narrative: str,
    prompts: dict[str, str],
    lang: str,
) -> list[dict]:
    """为单个论文章节生成 section + content slides。"""
    template = prompts.get("generate_section")
    if not template:
        raise ValueError("prompt.json is missing generate_section prompt")

    section_en = section_entry["title"]
    section_label = format_section_label(section_en, lang)
    focus_list = _format_focus_list(section_entry.get("focus") or [])
    pages_hint = section_entry.get("pages", "1 page")
    min_pages = _min_pages(pages_hint)

    user_prompt = template.format(
        section_label=section_label,
        section_en=section_en,
        focus_list=focus_list,
        pages_hint=pages_hint,
        min_pages=min_pages,
        paper_notes=paper_notes,
        figure_catalog=figure_catalog,
        table_catalog=table_catalog,
        narrative=narrative[:2000] if narrative else "(none)",
    )
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": user_prompt},
    ]
    response = _backend_chat(
        backend,
        messages,
        temperature=0.35,
        num_ctx=SYNTHESIS_NUM_CTX,
        timeout=SYNTHESIS_TIMEOUT,
    )
    data = extract_json(response)
    slides = data.get("slides") or []
    if not slides:
        return [
            {"type": "section", "title": section_label},
            {
                "type": "content",
                "title": section_label,
                "section": section_label,
                "bullets": [],
            },
        ]
    return slides


def ensure_title_slide(
    slides: list[dict],
    title: str,
    authors: str,
) -> list[dict]:
    """保证 decks 以 type=title 页开头。"""
    if slides and slides[0].get("type") == "title":
        slides[0]["title"] = title or slides[0].get("title") or "Research Paper"
        if authors:
            slides[0]["subtitle"] = authors
        return slides
    return [
        {
            "type": "title",
            "title": title or "Research Paper",
            "subtitle": authors or "",
        },
        *slides,
    ]


def fill_missing_titles(slides: list[dict], lang: str) -> list[dict]:
    """为 content 页补全空 title（用 section 或默认占位）。"""
    default = "要点" if lang.startswith("zh") else "Key Points"
    current_section = ""
    for slide in slides:
        if slide.get("type") == "section":
            current_section = slide.get("title") or ""
        if slide.get("type") != "content":
            continue
        if not (slide.get("title") or "").strip():
            slide["title"] = slide.get("section") or current_section or default
        if not slide.get("section") and current_section:
            slide["section"] = current_section
    return slides


def drop_empty_content_slides(slides: list[dict]) -> list[dict]:
    """移除无 bullets 且无 figure/table 的空 content 页。"""
    result: list[dict] = []
    for slide in slides:
        if slide.get("type") != "content":
            result.append(slide)
            continue
        has_body = bool(slide.get("bullets")) or slide.get("figure") or slide.get("table")
        if has_body:
            result.append(slide)
    return result


def build_deck(
    backend: LLMBackend,
    paper: PaperContent,
    prompts: dict[str, str],
    sections: list[dict],
    lang: str,
) -> dict:
    """
    多阶段构建完整 slide_data dict（含 title / authors / slides）。
    """
    structure_prompt = format_structure_for_prompt(sections, lang=lang)
    figure_catalog = build_figure_catalog(paper.figures, lang=lang)
    table_catalog = build_table_catalog(paper.tables, lang=lang)

    structure_prompt = format_structure_for_prompt(sections, lang=lang)

    print("      Phase 1/3: Deep-read paper (chunked)...")
    paper_notes = deep_read_paper(backend, paper.text, prompts)

    print("      Phase 2/3: Extract metadata...")
    meta = extract_metadata(backend, paper, paper_notes, prompts)
    print(f"      Title: {meta['title'][:60]}...")

    narrative = ""
    if prompts.get("build_narrative"):
        from paper2ppt.paper_chunk import build_narrative

        print("      Building narrative outline...")
        narrative = build_narrative(backend, paper_notes, prompts, structure_prompt)

    all_slides: list[dict] = []
    print("      Phase 3/3: Generate slides section-by-section...")
    with tqdm(sections, desc="      Section slides", unit="sec", dynamic_ncols=True) as bar:
        for entry in bar:
            label = format_section_label(entry["title"], lang)
            bar.set_postfix_str(label, refresh=False)
            part = generate_section_slides(
                backend,
                entry,
                paper_notes=paper_notes,
                figure_catalog=figure_catalog,
                table_catalog=table_catalog,
                narrative=narrative,
                prompts=prompts,
                lang=lang,
            )
            all_slides.extend(part)

    all_slides = ensure_title_slide(all_slides, meta["title"], meta["authors"])
    all_slides = fill_missing_titles(all_slides, lang)
    all_slides = drop_empty_content_slides(all_slides)
    all_slides = polish_slides(all_slides, lang=lang)
    all_slides = fill_missing_titles(all_slides, lang)

    return {
        "title": meta["title"],
        "authors": meta["authors"],
        "slides": all_slides,
    }
