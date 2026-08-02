"""
主流程编排模块。
"""

from __future__ import annotations

from pathlib import Path

from paper2ppt.deck_builder import (
    build_deck,
    ensure_overview_slide,
    ensure_title_slide,
    fill_missing_titles,
)
from paper2ppt.figure_assign import assign_figures_to_slides
from paper2ppt.llm.base import LLMBackend
from paper2ppt.pdf_parser import (
    PaperContent,
    cleanup_assets,
    extract_pdf,
)
from paper2ppt.ppt_generator import count_embedded_images, generate_ppt
from paper2ppt.prompts import DEFAULT_LANG, load_prompts, normalize_lang
from paper2ppt.slide_balance import balance_slides
from paper2ppt.slide_normalize import flatten_sections
from paper2ppt.table_assign import assign_tables_to_slides
from paper2ppt.paper_structure import (
    DEFAULT_STRUCTURE_PATH,
    audit_slide_structure,
    load_paper_structure,
)
DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "template" / "index.pptx"


def run_pipeline(
    pdf_path: str | Path,
    backend: LLMBackend,
    *,
    output_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    structure_path: str | Path | None = None,
    lang: str = DEFAULT_LANG,
    template_path: str | Path | None = None,
) -> Path:
    lang = normalize_lang(lang)
    pdf_path = Path(pdf_path)
    if output_path is None:
        output_path = pdf_path.with_suffix(".pptx")
    output_path = Path(output_path)

    template = Path(template_path) if template_path else DEFAULT_TEMPLATE
    assets_dir: Path | None = None

    try:
        print(f"[1/4] Parsing PDF and extracting figures: {pdf_path}")
        paper = extract_pdf(pdf_path)
        assets_dir = paper.assets_dir
        if paper.figures:
            names = ", ".join(f.figure_id for f in paper.figures)
            print(f"      Extracted figures: {names}")
        else:
            print("      No figure captions found; text-only slides")
        if paper.tables:
            tnames = ", ".join(t.table_id for t in paper.tables)
            print(f"      Extracted tables: {tnames}")

        print(f"[2/4] Building deck section-by-section (model={getattr(backend, 'model', 'unknown')}, lang={lang})...")
        slide_data = _build_deck(
            backend,
            paper,
            prompt_path,
            lang,
            structure_path=structure_path,
        )

        slides = flatten_sections(slide_data.get("slides", []))
        slide_data["slides"] = fill_missing_titles(slides, lang)
        slide_data["slides"] = ensure_title_slide(
            slide_data["slides"],
            slide_data.get("title", ""),
            slide_data.get("authors", ""),
        )
        slide_data["slides"] = ensure_overview_slide(slide_data["slides"], lang)
        raw_count = len(slide_data.get("slides", []))
        assigned = assign_figures_to_slides(
            slide_data["slides"],
            paper.figures,
            table_reserve=len(paper.tables),
        )
        table_assigned = assign_tables_to_slides(slide_data["slides"], paper.tables)
        assigned, table_assigned = _limit_visual_density(slide_data["slides"])
        print(f"      Selected figures on {assigned} slide(s)")
        if paper.tables:
            print(f"      Selected tables on {table_assigned} slide(s)")

        slide_data["slides"] = balance_slides(slide_data["slides"], lang=lang)
        balanced_count = len(slide_data["slides"])
        if balanced_count > raw_count:
            print(f"      Length balance: {raw_count} slides -> {balanced_count} slides")

        print(f"[3/4] Generating PPT (template: {template}): {output_path}")
        figure_map = {fig.figure_id: fig.path for fig in paper.figures}
        table_map = {tbl.table_id: tbl for tbl in paper.tables}
        generate_ppt(
            slide_data,
            output_path,
            figures=figure_map,
            tables=table_map,
            template_path=template,
        )

        embedded = count_embedded_images(output_path)
        print(f"      Output contains {embedded} embedded image(s)")
        if paper.figures and embedded == 0:
            print("      Warning: figures were extracted but none appear in the PPT")

        print(f"[4/4] Done! Output: {output_path}")
        return output_path
    finally:
        if assets_dir:
            cleanup_assets(assets_dir)
            print(f"      Cleaned up temp dir: {assets_dir}")


def _limit_visual_density(
    slides: list[dict], *, max_figures: int = 10, max_tables: int = 5
) -> tuple[int, int]:
    """控制视觉素材密度，保留组会节奏，避免变成逐图/逐表翻译。"""
    figure_count = 0
    table_count = 0
    for slide in slides:
        if slide.get("type") != "content":
            continue
        if slide.get("figure"):
            if figure_count < max_figures:
                figure_count += 1
            else:
                slide.pop("figure", None)
        if slide.get("table") or slide.get("table_data"):
            if table_count < max_tables:
                table_count += 1
            else:
                slide.pop("table", None)
                slide.pop("table_data", None)
    return figure_count, table_count


def _build_deck(
    backend: LLMBackend,
    paper: PaperContent,
    prompt_path: str | Path | None,
    lang: str,
    *,
    structure_path: str | Path | None = None,
) -> dict:
    """
    多阶段构建幻灯片：分片读透 → 逐章生成 → 结构审计。
    """
    prompts = load_prompts(prompt_path, lang=lang)

    structure_file = Path(structure_path) if structure_path else DEFAULT_STRUCTURE_PATH
    sections = load_paper_structure(structure_file)
    print(f"      Loaded deck structure: {structure_file.name} ({len(sections)} sections)")

    slide_data = build_deck(
        backend,
        paper,
        prompts,
        sections,
        lang,
    )

    structure_warnings = audit_slide_structure(
        slide_data.get("slides", []), sections, lang=lang
    )
    if structure_warnings:
        for msg in structure_warnings:
            print(f"      Warning: {msg}")
    else:
        print(f"      Structure check passed ({len(sections)} sections present)")

    return slide_data
