"""
主流程编排模块。
"""

from __future__ import annotations

from pathlib import Path

from paper2ppt.figure_assign import assign_figures_to_slides
from paper2ppt.llm.base import LLMBackend
from paper2ppt.pdf_parser import (
    PaperContent,
    build_figure_catalog,
    cleanup_assets,
    extract_pdf,
)
from paper2ppt.ppt_generator import count_embedded_images, generate_ppt
from paper2ppt.prompts import DEFAULT_LANG, extract_json, load_prompts, normalize_lang
from paper2ppt.slide_balance import balance_slides
from paper2ppt.slide_normalize import flatten_sections

MAX_PAPER_CHARS = 120_000
DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "template" / "index.pptx"


def run_pipeline(
    pdf_path: str | Path,
    backend: LLMBackend,
    *,
    output_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
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
        print(f"[1/4] 解析 PDF 并提取 Figure: {pdf_path}")
        paper = extract_pdf(pdf_path)
        assets_dir = paper.assets_dir
        if paper.figures:
            names = ", ".join(f.figure_id for f in paper.figures)
            print(f"      已提取 Figure: {names}")
        else:
            print("      未识别到 Figure 图注，将仅生成文字幻灯片")

        print(f"[2/4] 调用 LLM 分析论文 (model={getattr(backend, 'model', 'unknown')}, lang={lang})...")
        slide_data = _analyze_paper(backend, paper, prompt_path, lang)

        slides = flatten_sections(slide_data.get("slides", []))
        raw_count = len(slide_data.get("slides", []))
        slide_data["slides"] = balance_slides(slides, lang=lang)
        balanced_count = len(slide_data["slides"])
        if len(slides) < raw_count:
            print(f"      章节合并：{raw_count} 项 → {len(slides)} 页（已去掉独立 section 页）")
        if balanced_count > len(slides):
            print(f"      篇幅平衡：{len(slides)} 页 → {balanced_count} 页（已拆分过长内容）")

        assigned = assign_figures_to_slides(slide_data["slides"], paper.figures)
        print(f"      已为 {assigned} 页幻灯片匹配 Figure 插图")

        print(f"[3/4] 生成 PPT (模板: {template}): {output_path}")
        figure_map = {fig.figure_id: fig.path for fig in paper.figures}
        generate_ppt(
            slide_data,
            output_path,
            figures=figure_map,
            template_path=template,
        )

        embedded = count_embedded_images(output_path)
        print(f"      输出文件含 {embedded} 个内嵌图片资源")
        if paper.figures and embedded == 0:
            print("      警告: 已提取 Figure 但 PPT 中未检测到图片，请检查匹配逻辑")

        print(f"[4/4] 完成! 输出文件: {output_path}")
        return output_path
    finally:
        if assets_dir:
            cleanup_assets(assets_dir)
            print(f"      已清理临时目录: {assets_dir}")


def _analyze_paper(
    backend: LLMBackend,
    paper: PaperContent,
    prompt_path: str | Path | None,
    lang: str,
) -> dict:
    prompts = load_prompts(prompt_path, lang=lang)
    paper_text = paper.text

    if len(paper_text) > MAX_PAPER_CHARS:
        paper_text = paper_text[:MAX_PAPER_CHARS] + "\n\n[... 论文内容已截断 ...]"

    figure_catalog = build_figure_catalog(paper.figures, lang=lang)
    user_prompt = prompts["analyze_paper"].format(
        paper_text=paper_text,
        figure_catalog=figure_catalog,
    )
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": user_prompt},
    ]

    response = backend.chat(messages)
    slide_data = extract_json(response)

    if not slide_data.get("title") and paper.title:
        slide_data["title"] = paper.title

    return slide_data
