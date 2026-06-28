"""
PDF 解析模块 — 使用 PyMuPDF 提取论文文本与 Figure 插图。

Figure 提取策略：
  1. 逐页扫描 Figure/Fig./图 N 形式的图注
  2. 在同一页上，将图注与图注上方最近的大尺寸嵌入图匹配
  3. 若无合适嵌入图，则裁剪图注上方的页面区域作为 figureN.png
  4. 图片保存为 figure1.png, figure2.png … 供 LLM 在 slides 中引用

临时资源目录：<pdf目录>/.<stem>_assets/，pipeline 结束后自动清理。
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import fitz  # pymupdf

# 行首 Figure/Fig./图 N，且序号后紧跟冒号或句点（排除 "Fig. 4 depicts..." 类正文句）
FIGURE_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.?)\s*(\d+)\s*[:\.]|^图\s*(\d+)\s*[：:]",
    re.IGNORECASE,
)

# 嵌入图最小面积（平方点），过滤 logo/图标
MIN_IMAGE_AREA = 8_000


@dataclass
class PaperFigure:
    """从论文中提取的单张 Figure。"""

    figure_id: str   # 引用 id，如 "figure1"
    number: int      # 图序号
    path: Path       # 落盘路径，如 figure1.png
    page: int        # 所在页码（从 1 开始）
    caption: str     # 图注原文（截断）


@dataclass
class PaperContent:
    """PDF 解析结果。"""

    text: str
    title: str = ""
    figures: list[PaperFigure] = field(default_factory=list)
    assets_dir: Path | None = None


def default_assets_dir(pdf_path: Path) -> Path:
    """返回 PDF 对应的临时资源目录路径。"""
    return pdf_path.parent / f".{pdf_path.stem}_assets"


def cleanup_assets(assets_dir: str | Path | None) -> None:
    """删除临时资源目录（如 .dsd_assets）。"""
    if not assets_dir:
        return
    path = Path(assets_dir)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def build_figure_catalog(figures: list[PaperFigure], lang: str = "zh_cn") -> str:
    """
    生成供 LLM 参考的 Figure 清单文本。

    LLM 应在相关幻灯片的 "figure" 字段填入 figure_id。
    """
    if not figures:
        if lang == "en_us":
            return "No figures were extracted from the PDF. Do not set the \"figure\" field."
        return "未能从 PDF 中提取到 Figure。请勿填写 \"figure\" 字段。"

    lines: list[str] = []
    for fig in sorted(figures, key=lambda f: f.number):
        if lang == "en_us":
            lines.append(
                f'- {fig.figure_id} (page {fig.page}): {fig.caption[:200]}'
            )
        else:
            lines.append(
                f'- {fig.figure_id}（第 {fig.page} 页）: {fig.caption[:200]}'
            )
    return "\n".join(lines)


def extract_pdf(pdf_path: str | Path, output_dir: str | Path | None = None) -> PaperContent:
    """
    解析 PDF：提取全文、识别 Figure 并截图/导出为 figureN.png。

    参数:
        pdf_path  : PDF 文件路径。
        output_dir: 临时资源目录；默认 .<stem>_assets/

    返回:
        PaperContent，含 figures 列表与 assets_dir 路径（供后续清理）。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF 文件: {pdf_path}")

    assets_dir = Path(output_dir) if output_dir else default_assets_dir(pdf_path)
    # 清空旧缓存，避免 figure 编号与旧文件混淆
    cleanup_assets(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pages_text.append(page.get_text())

    figures = _extract_figures(doc, assets_dir)
    doc.close()

    full_text = "\n\n".join(pages_text).strip()
    title = _guess_title(pages_text[0] if pages_text else "")

    return PaperContent(
        text=full_text,
        title=title,
        figures=figures,
        assets_dir=assets_dir,
    )


def _extract_figures(doc: fitz.Document, output_dir: Path) -> list[PaperFigure]:
    """扫描全文，按图注序号提取并保存 figureN.png。"""
    # figure_number -> (page, caption, xref_or_none, clip_rect_or_none)
    caption_entries: dict[int, tuple[int, str, fitz.Rect]] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        for fig_num, bbox, caption in _find_figure_captions(page):
            if fig_num not in caption_entries:
                caption_entries[fig_num] = (page_num + 1, caption, bbox)

    figures: list[PaperFigure] = []
    used_xrefs: set[int] = set()

    for fig_num in sorted(caption_entries):
        page_num, caption, caption_rect = caption_entries[fig_num]
        page = doc[page_num - 1]
        figure_id = f"figure{fig_num}"
        out_path = output_dir / f"{figure_id}.png"

        saved = _save_matched_image(
            doc, page, caption_rect, out_path, used_xrefs
        )
        if not saved:
            saved = _save_page_clip(page, caption_rect, out_path)

        if saved:
            figures.append(
                PaperFigure(
                    figure_id=figure_id,
                    number=fig_num,
                    path=out_path,
                    page=page_num,
                    caption=caption,
                )
            )

    return figures


def _find_figure_captions(page: fitz.Page) -> list[tuple[int, fitz.Rect, str]]:
    """在一页中查找所有 Figure 图注，返回 (序号, bbox, 图注文本)。"""
    results: list[tuple[int, fitz.Rect, str]] = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if not line_text:
                continue
            match = FIGURE_CAPTION_RE.match(line_text)
            if not match:
                continue
            num_str = match.group(1) or match.group(2)
            fig_num = int(num_str)
            bbox = fitz.Rect(line["bbox"])
            # 合并后续行作为完整图注（最多 3 行）
            full_caption = line_text
            results.append((fig_num, bbox, full_caption))

    return results


def _page_image_candidates(page: fitz.Page) -> list[tuple[int, fitz.Rect, float]]:
    """收集页面上所有足够大的嵌入图：(xref, rect, area)。"""
    candidates: list[tuple[int, fitz.Rect, float]] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            area = rect.width * rect.height
            if area >= MIN_IMAGE_AREA:
                candidates.append((xref, rect, area))
    return candidates


def _save_matched_image(
    doc: fitz.Document,
    page: fitz.Page,
    caption_rect: fitz.Rect,
    out_path: Path,
    used_xrefs: set[int],
) -> bool:
    """
    在图注上方寻找最匹配的嵌入图并保存为 PNG。

    优先选择：位于图注上方、面积最大、尚未被其他 Figure 使用的图片。
    """
    candidates = _page_image_candidates(page)
    if not candidates:
        return False

    above = [
        (xref, rect, area)
        for xref, rect, area in candidates
        if rect.y1 <= caption_rect.y0 + 20 and xref not in used_xrefs
    ]
    if not above:
        # 放宽：允许与图注少量重叠，取面积最大且未使用的
        above = [
            (xref, rect, area)
            for xref, rect, area in candidates
            if xref not in used_xrefs
        ]

    if not above:
        return False

    xref, rect, _ = max(above, key=lambda item: item[2])
    try:
        base_image = doc.extract_image(xref)
    except Exception:
        return False
    if not base_image or not base_image.get("image"):
        return False

    _write_png(base_image["image"], out_path)
    used_xrefs.add(xref)
    return True


def _save_page_clip(page: fitz.Page, caption_rect: fitz.Rect, out_path: Path) -> bool:
    """裁剪图注上方的页面区域作为 Figure（无嵌入图时的 fallback）。"""
    page_rect = page.rect
    margin = 36
    top = page_rect.y0 + margin
    bottom = caption_rect.y0 - 8
    left = page_rect.x0 + margin
    right = page_rect.x1 - margin

    if bottom - top < 80 or right - left < 80:
        return False

    clip = fitz.Rect(left, top, right, bottom)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
        out_path.write_bytes(pix.tobytes("png"))
        return True
    except Exception:
        return False


def _write_png(image_bytes: bytes, out_path: Path) -> None:
    """将图片字节写入 PNG 文件（必要时经 Pillow 转格式）。"""
    try:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out_path, format="PNG")
    except Exception:
        out_path.write_bytes(image_bytes)


def _guess_title(first_page: str) -> str:
    """从首页文本启发式猜测论文标题。"""
    for line in first_page.splitlines():
        line = line.strip()
        if len(line) > 10 and not line.isdigit():
            return line[:200]
    return ""
