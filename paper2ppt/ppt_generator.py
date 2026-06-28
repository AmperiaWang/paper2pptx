"""
PowerPoint 生成模块 — 基于 python-pptx，支持自定义模板母版。

文字颜色、字体等样式继承自模板，仅按需调整字号与段落间距。
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

from paper2ppt.slide_balance import estimate_content_font_size, estimate_title_font_size

LAYOUT_TITLE = 0
LAYOUT_CONTENT = 1
LAYOUT_PICTURE = 8

TITLE_TYPES = frozenset({
    PP_PLACEHOLDER.TITLE,
    PP_PLACEHOLDER.CENTER_TITLE,
})
BODY_TYPES = frozenset({
    PP_PLACEHOLDER.BODY,
    PP_PLACEHOLDER.OBJECT,
})
SKIP_TYPES = frozenset({
    PP_PLACEHOLDER.PICTURE,
    PP_PLACEHOLDER.DATE,
    PP_PLACEHOLDER.FOOTER,
    PP_PLACEHOLDER.SLIDE_NUMBER,
    PP_PLACEHOLDER.HEADER,
})


def generate_ppt(
    slide_data: dict,
    output_path: str | Path,
    figures: dict[str, Path] | None = None,
    template_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    template_path = Path(template_path) if template_path else _default_template()

    if not template_path.exists():
        raise FileNotFoundError(f"找不到 PPT 模板: {template_path}")

    figure_bytes = _load_figure_bytes(figures or {})

    prs = Presentation(str(template_path))
    _remove_existing_slides(prs)
    picture_layout_idx = _find_picture_layout(prs)

    inserted = 0
    for slide_info in slide_data.get("slides", []):
        slide_type = slide_info.get("type", "content")

        if slide_type == "title":
            _add_title_slide(prs, slide_info, slide_data)
        elif slide_type == "section":
            # 规范化后不应再出现 section；若出现则当作仅标题内容页处理
            _add_content_slide(prs, {
                "title": slide_info.get("title", ""),
                "bullets": slide_info.get("bullets", []),
            }, image_data=None, picture_layout_idx=picture_layout_idx)
        else:
            figure_id = slide_info.get("figure")
            image_data = figure_bytes.get(figure_id) if figure_id else None
            _add_content_slide(
                prs,
                slide_info,
                image_data=image_data,
                picture_layout_idx=picture_layout_idx,
            )
            if image_data:
                inserted += 1

    prs.save(str(output_path))
    print(f"      已嵌入 {inserted} 张 Figure 图片，共 {len(prs.slides)} 页")
    return output_path


def count_embedded_images(pptx_path: str | Path) -> int:
    with zipfile.ZipFile(pptx_path) as zf:
        return sum(1 for name in zf.namelist() if name.startswith("ppt/media/"))


def _default_template() -> Path:
    return Path(__file__).resolve().parent.parent / "template" / "index.pptx"


def _remove_existing_slides(prs: Presentation) -> None:
    slide_ids = list(prs.slides._sldIdLst)
    for sld_id in slide_ids:
        r_id = sld_id.rId
        prs.part.drop_rel(r_id)
        prs.slides._sldIdLst.remove(sld_id)


def _load_figure_bytes(figures: dict[str, Path]) -> dict[str, bytes]:
    loaded: dict[str, bytes] = {}
    for fid, path in figures.items():
        path = Path(path)
        if path.is_file():
            loaded[fid] = path.read_bytes()
    return loaded


def _find_picture_layout(prs: Presentation) -> int:
    for idx, layout in enumerate(prs.slide_layouts):
        for ph in layout.placeholders:
            if ph.placeholder_format.type == PP_PLACEHOLDER.PICTURE:
                return idx
    return min(LAYOUT_PICTURE, len(prs.slide_layouts) - 1)


def _find_title_placeholder(slide):
    """查找标题占位符（支持 TITLE 与 CENTER_TITLE）。"""
    if slide.shapes.title is not None:
        return slide.shapes.title
    for ph in slide.placeholders:
        if ph.placeholder_format.type in TITLE_TYPES:
            return ph
    return None


def _find_body_placeholder(slide):
    """
    查找正文占位符。

    模板「标题和内容」版式的正文类型为 OBJECT(7)，不是 BODY(2)；
    必须按类型匹配，绝不能回退到第一个非图片占位符（那会是标题）。
    """
    for ph in slide.placeholders:
        if ph.placeholder_format.type in BODY_TYPES:
            return ph
    return None


def _find_subtitle_placeholder(slide):
    for ph in slide.placeholders:
        if ph.placeholder_format.type == PP_PLACEHOLDER.SUBTITLE:
            return ph
    return None


def _configure_text_frame(text_frame) -> None:
    text_frame.word_wrap = True
    try:
        text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    except Exception:
        pass


def _set_font_size(shape, size: int) -> None:
    """仅调整字号，不覆盖模板颜色/字体。"""
    if shape is None:
        return
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(size)


def _add_title_slide(prs: Presentation, slide_info: dict, slide_data: dict):
    layout = prs.slide_layouts[min(LAYOUT_TITLE, len(prs.slide_layouts) - 1)]
    slide = prs.slides.add_slide(layout)

    title_text = slide_info.get("title") or slide_data.get("title", "Research Paper")
    title_ph = _find_title_placeholder(slide)
    if title_ph is not None:
        title_ph.text = title_text
        title_size = estimate_title_font_size(title_text, default=36)
        _configure_text_frame(title_ph.text_frame)
        _set_font_size(title_ph, title_size)

    subtitle_ph = _find_subtitle_placeholder(slide)
    if subtitle_ph is not None:
        parts = []
        if slide_info.get("subtitle"):
            parts.append(slide_info["subtitle"])
        if slide_data.get("authors"):
            parts.append(slide_data["authors"])
        subtitle_ph.text = "\n".join(parts) if parts else ""
        _configure_text_frame(subtitle_ph.text_frame)
        _set_font_size(subtitle_ph, 18)

    return slide


def _add_content_slide(
    prs: Presentation,
    slide_info: dict,
    *,
    image_data: bytes | None,
    picture_layout_idx: int,
):
    has_figure = image_data is not None
    layout_idx = picture_layout_idx if has_figure else min(LAYOUT_CONTENT, len(prs.slide_layouts) - 1)
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])

    bullets = slide_info.get("bullets", [])
    title_text = slide_info.get("title", "")

    title_ph = _find_title_placeholder(slide)
    if title_ph is not None:
        title_ph.text = title_text
        title_size = estimate_title_font_size(title_text, default=28)
        _configure_text_frame(title_ph.text_frame)
        _set_font_size(title_ph, title_size)

    body_ph = _find_body_placeholder(slide)
    font_size = estimate_content_font_size(bullets, has_figure)

    if body_ph is not None:
        tf = body_ph.text_frame
        tf.clear()
        _configure_text_frame(tf)

        for i, bullet in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = bullet
            p.level = 0
            p.space_after = Pt(4)
            for run in p.runs:
                run.font.size = Pt(font_size)

    if image_data:
        _insert_picture(slide, image_data)

    return slide


def _insert_picture(slide, image_data: bytes) -> bool:
    stream = BytesIO(image_data)
    for ph in slide.placeholders:
        if ph.placeholder_format.type == PP_PLACEHOLDER.PICTURE:
            ph.insert_picture(stream)
            return True
    slide.shapes.add_picture(stream, Inches(7.0), Inches(1.5), width=Inches(5.5))
    return True
