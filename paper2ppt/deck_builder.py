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

import json
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

PLAN_DECK_PROMPT = """
你是学术汇报的内容策划师。先规划整套 PPT 的每一页，避免重复和遗漏。

## 结构与页数
{paper_structure}
## 论文笔记
{paper_notes}
## 可用图
{figure_catalog}
## 可用表
{table_catalog}

只返回 JSON：
{{"sections":[{{"section":"Introduction","pages":[{{"title":"结论式标题","purpose":"本页回答的问题","evidence":["事实或数字"],"visual":"figure1/table1/none"}}]}}]}}

严格按每章 pages 的上限规划。每页一个中心结论。Methods 按总览、核心模块、训练、推理拆页；Experiment 按设置、主结果、消融、效率和案例拆页。优先分配高相关 Figure/Table，同一素材只用一次，图表不共页。禁止编造。
""".strip()


def _backend_chat(
    backend: LLMBackend,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    num_ctx: int | None = None,
    timeout: int = 600,
    json_mode: bool = False,
) -> str:
    if isinstance(backend, OllamaBackend):
        return backend.chat(
            messages,
            temperature=temperature,
            num_ctx=num_ctx,
            timeout=timeout,
            json_mode=json_mode,
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
    authors = (paper.authors or "").strip()

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
                    json_mode=True,
                )
            )
            # Front-page extraction is authoritative; LLM is fallback only.
            title = (title or data.get("title") or "Research Paper").strip()
            authors = (authors or data.get("authors") or "").strip()
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
    section_source: str,
    figure_catalog: str,
    table_catalog: str,
    narrative: str,
    section_plan: str,
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
        narrative=narrative[:5000] if narrative else "(none)",
        section_plan=section_plan or "(未提供，请按 focus 自行细化)",
    )
    user_prompt += f"""

## 全局规划分配给本章的页面（必须逐页落实）
{section_plan or '(无)'}

## 内容密度与视觉规则
- 每个规划页面都要生成，不得把多个主题缩成一页
- 每页 4–6 条互不重复的要点：背景/问题、具体设计、机制原因、证据/数字、含义
- 每条优先 35–90 个中文字，保留方法名、公式变量、数据集、baseline 和定量结果
- 有 visual 分配时必须填对应 figure/table 字段；图表不共页
- 标题必须是该页的具体结论，不能只写「方法」「实验」或「结果」

## 本章对应的论文原文（权威事实源，优先级高于分片笔记）
{section_source}
"""
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
        json_mode=True,
    )
    try:
        data = extract_json(response)
    except ValueError as exc:
        # A model can occasionally ignore JSON mode.  Retry once with the
        # parser error in context instead of aborting the whole long pipeline.
        retry_messages = [
            *messages,
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    "Your previous answer was not a valid JSON object "
                    f"({exc}). Return only the requested JSON object now."
                ),
            },
        ]
        response = _backend_chat(
            backend,
            retry_messages,
            temperature=0.1,
            num_ctx=SYNTHESIS_NUM_CTX,
            timeout=SYNTHESIS_TIMEOUT,
            json_mode=True,
        )
        data = extract_json(response)
    slides = data.get("slides") or []
    issues = _section_quality_issues(slides, min_pages)
    if issues:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    "上一版内容过于简略，问题为："
                    + "；".join(issues)
                    + "。请重写完整 JSON，保留所有规划页；每页 4–6 条不重复的具体要点，"
                    "补足机制、步骤、实验设置、定量证据及其含义。"
                    "只能使用论文笔记中已有事实；不得为了凑数编造或重复。"
                ),
            },
        ]
        try:
            repaired = _backend_chat(
                backend,
                repair_messages,
                temperature=0.2,
                num_ctx=SYNTHESIS_NUM_CTX,
                timeout=SYNTHESIS_TIMEOUT,
                json_mode=True,
            )
            repaired_slides = extract_json(repaired).get("slides") or []
            repaired_count = len(
                [s for s in repaired_slides if s.get("type") == "content"]
            )
            if repaired_count >= min_pages:
                slides = repaired_slides
        except (RuntimeError, ValueError):
            # Keep the first valid response if the optional density repair fails.
            pass
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
    if not any(slide.get("type") == "section" for slide in slides):
        slides.insert(0, {"type": "section", "title": section_label})
    return _limit_section_pages(slides, min_pages)


def _limit_section_pages(slides: list[dict], target_pages: int) -> list[dict]:
    """严格执行全局页数规划，防止模型自行膨胀章节。"""
    result: list[dict] = []
    content_count = 0
    for slide in slides:
        if slide.get("type") == "section":
            if not any(item.get("type") == "section" for item in result):
                result.append(slide)
            continue
        if slide.get("type") != "content":
            continue
        if content_count >= target_pages:
            continue
        result.append(slide)
        content_count += 1
    return result


def _section_quality_issues(slides: list[dict], min_pages: int) -> list[str]:
    """检测章节是否因小模型偷懒而过于简略。"""
    content = [slide for slide in slides if slide.get("type") == "content"]
    issues: list[str] = []
    if len(content) < min_pages:
        issues.append(f"只有 {len(content)} 页，需要至少 {min_pages} 页")
    sparse = [
        slide
        for slide in content
        if len(slide.get("bullets") or []) < 4
        or sum(len(str(b)) for b in slide.get("bullets") or []) < 140
    ]
    if sparse:
        issues.append(f"{len(sparse)} 页的要点数或信息量不足")
    return issues


def plan_deck(
    backend: LLMBackend,
    *,
    paper_notes: str,
    figure_catalog: str,
    table_catalog: str,
    structure_prompt: str,
    prompts: dict[str, str],
    sections: list[dict],
) -> dict[str, list[dict]]:
    """先全局分配每页主题与视觉素材，再逐章生成正文。"""
    template = prompts.get("plan_deck") or PLAN_DECK_PROMPT
    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": template.format(
                paper_structure=structure_prompt,
                paper_notes=paper_notes,
                figure_catalog=figure_catalog,
                table_catalog=table_catalog,
            ),
        },
    ]
    data = extract_json(
        _backend_chat(
            backend,
            messages,
            temperature=0.15,
            num_ctx=SYNTHESIS_NUM_CTX,
            timeout=SYNTHESIS_TIMEOUT,
            json_mode=True,
        )
    )
    result: dict[str, list[dict]] = {}
    for section in data.get("sections") or []:
        name = str(section.get("section") or "").strip().lower()
        pages = section.get("pages") or []
        if name and isinstance(pages, list):
            result[name] = pages
    return _complete_deck_plan(result, sections)


def _complete_deck_plan(
    plan: dict[str, list[dict]], sections: list[dict]
) -> dict[str, list[dict]]:
    """用结构中的 focus 补齐模型漏掉的规划页。"""
    completed: dict[str, list[dict]] = {}
    for section in sections:
        name = section["title"].strip().lower()
        target = _min_pages(section.get("pages", "1 page"))
        pages = list(plan.get(name) or [])[:target]
        focus = section.get("focus") or [section["title"]]
        while len(pages) < target:
            index = len(pages)
            question = focus[index % len(focus)]
            pages.append(
                {
                    "title": f"{section['title']} 主题 {index + 1}",
                    "purpose": question,
                    "evidence": ["从论文笔记中提取相关机制、步骤和定量证据"],
                    "visual": "choose the most relevant unused figure/table, or none",
                }
            )
        completed[name] = pages
    return completed


def _section_plan_text(plan: dict[str, list[dict]], section_title: str) -> str:
    key = section_title.strip().lower()
    pages = plan.get(key)
    if not pages:
        pages = next(
            (value for name, value in plan.items() if key in name or name in key),
            [],
        )
    return json.dumps(pages, ensure_ascii=False, indent=2) if pages else ""


def _section_source_text(paper_text: str, section_title: str) -> str:
    """从带页码的正文中为汇报章节选取直接原文依据。"""
    page_parts = re.split(r"(?=\[PDF page \d+\])", paper_text)
    selected: list[str] = []
    key = section_title.lower()
    for part in page_parts:
        match = re.match(r"\[PDF page (\d+)\]", part)
        page = int(match.group(1)) if match else 0
        if key == "introduction":
            keep = 1 <= page <= 4
        elif key == "related work":
            keep = 34 <= page <= 36
        elif key == "methods":
            keep = 5 <= page <= 16 or 30 <= page <= 33
        elif key in {"experiment", "experiments"}:
            keep = 17 <= page <= 33
        elif key == "conclusion":
            keep = 37 <= page <= 39
        else:
            keep = False
        if keep:
            selected.append(part)
    source = "".join(selected).strip()
    if len(source) > 42_000:
        source = source[:30_000] + "\n\n[...middle omitted...]\n\n" + source[-12_000:]
    return source or paper_text[:24_000]


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


def ensure_overview_slide(slides: list[dict], lang: str) -> list[dict]:
    """在封面后加入汇报地图，建立组会讲述预期。"""
    overview_titles = {"汇报地图", "talk roadmap"}
    if any((slide.get("title") or "").lower() in overview_titles for slide in slides):
        return slides
    section_titles = [
        slide.get("title")
        for slide in slides
        if slide.get("type") == "section" and slide.get("title")
    ]
    if not section_titles:
        return slides
    overview = {
        "type": "content",
        "title": "汇报地图" if lang.startswith("zh") else "Talk Roadmap",
        "bullets": [
            f"**{index:02d}**  {name}" for index, name in enumerate(section_titles, 1)
        ],
    }
    insert_at = 1 if slides and slides[0].get("type") == "title" else 0
    return [*slides[:insert_at], overview, *slides[insert_at:]]


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

    print("      Planning slide topics and visual allocation...")
    deck_plan = plan_deck(
        backend,
        paper_notes=paper_notes,
        figure_catalog=figure_catalog,
        table_catalog=table_catalog,
        structure_prompt=structure_prompt,
        prompts=prompts,
        sections=sections,
    )
    if deck_plan:
        planned_pages = sum(len(pages) for pages in deck_plan.values())
        print(f"      Planned {planned_pages} content slides with figure/table allocation")

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
                section_source=_section_source_text(paper.text, entry["title"]),
                figure_catalog=figure_catalog,
                table_catalog=table_catalog,
                narrative=narrative,
                section_plan=_section_plan_text(deck_plan, entry["title"]),
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
