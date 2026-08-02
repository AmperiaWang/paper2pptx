"""
PDF 表格提取 — 定位 Table N 图注，截取区域为图片，并尽力解析为二维单元格数据。

策略:
  1. 匹配 "Table N:" 图注行
  2. 向上搜索表格区域（多列对齐的文本块）
  3. 保存 tableN.png 截图
  4. 按 x 坐标聚类解析 rows，供 PPT 原生表格渲染
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz

TABLE_CAPTION_RE = re.compile(
    r"^(?:Table|TABLE)\s*(\d+)\s*[-–—:\.]",
    re.IGNORECASE,
)

# 表头 / 数据行识别
TABLE_HEADER_HINTS = re.compile(
    r"method|acc|accuracy|dataset|metric|参数|准确率|方法",
    re.IGNORECASE,
)
NUMERIC_CELL = re.compile(r"[\d.]+\s*[%±]|^\d+\.\d+")


@dataclass
class PaperTable:
    """从 PDF 提取的表格。"""

    table_id: str
    number: int
    page: int
    caption: str
    rows: list[list[str]]
    image_path: Path | None = None


def extract_tables(doc: fitz.Document, output_dir: Path) -> list[PaperTable]:
    """扫描 PDF，提取所有 Table N。"""
    entries: list[tuple[int, int, str, fitz.Rect]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        for fig_num, bbox, caption in _find_table_captions(page):
            entries.append((page_num, fig_num, caption, bbox))

    tables: list[PaperTable] = []
    for page_num, table_num, caption, caption_rect in sorted(entries, key=lambda e: (e[0], e[1])):
        page = doc[page_num]
        table_id = f"table{table_num}"
        image_path = output_dir / f"{table_id}.png"

        top = _find_table_top(page, caption_rect)
        clip = fitz.Rect(
            page.rect.x0 + 36,
            top,
            page.rect.x1 - 36,
            caption_rect.y0 - 4,
        )
        if clip.height < 30 or clip.width < 80:
            continue

        _save_clip(page, clip, image_path)
        rows = _parse_table_rows(page, clip)
        rows = _clean_table_rows(rows)

        tables.append(
            PaperTable(
                table_id=table_id,
                number=table_num,
                page=page_num + 1,
                caption=caption,
                rows=rows,
                image_path=image_path if image_path.is_file() else None,
            )
        )

    return tables


def build_table_catalog(tables: list[PaperTable], lang: str = "zh_cn") -> str:
    """生成供 LLM 参考的 Table 清单。"""
    if not tables:
        if lang == "en_us":
            return (
                'No tables were extracted. Use inline "table_data" only when necessary, '
                'or omit tables.'
            )
        return "未能从 PDF 中提取到 Table。若无表格可省略 \"table\" 字段。"

    lines: list[str] = []
    for tbl in sorted(tables, key=lambda t: t.number):
        preview = _rows_preview(tbl.rows)
        if lang == "en_us":
            lines.append(
                f'- {tbl.table_id} (page {tbl.page}): {tbl.caption[:120]}\n  Preview: {preview}'
            )
        else:
            lines.append(
                f'- {tbl.table_id}（第 {tbl.page} 页）: {tbl.caption[:120]}\n  预览: {preview}'
            )
    return "\n".join(lines)


def _rows_preview(rows: list[list[str]], max_rows: int = 3) -> str:
    if not rows:
        return "(image only)"
    parts = [" | ".join(r) for r in rows[:max_rows]]
    return "; ".join(parts)


def _find_table_captions(page: fitz.Page) -> list[tuple[int, fitz.Rect, str]]:
    results: list[tuple[int, fitz.Rect, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            match = TABLE_CAPTION_RE.match(text)
            if match:
                results.append((int(match.group(1)), fitz.Rect(line["bbox"]), text))
    return results


def _find_table_top(page: fitz.Page, caption_rect: fitz.Rect) -> float:
    """从图注向上找表格起始 y（表头行）。"""
    candidates: list[float] = []
    window_top = caption_rect.y0 - 200
    window_bottom = caption_rect.y0 - 6

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            y0 = line["bbox"][1]
            if y0 < window_top or y0 > window_bottom:
                continue
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text or len(text) > 120:
                continue
            # 表头行：短标签、Acc/Method 或多列数值标题
            if text in ("Acc", "Method") or TABLE_HEADER_HINTS.search(text):
                candidates.append(y0)
            elif len(spans) >= 3 and all(len(s.get("text", "").strip()) < 28 for s in spans):
                candidates.append(y0)

    if candidates:
        return max(page.rect.y0 + 28, min(candidates) - 4)
    return max(page.rect.y0 + 28, caption_rect.y0 - 95)


def _save_clip(page: fitz.Page, clip: fitz.Rect, out_path: Path) -> None:
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
        out_path.write_bytes(pix.tobytes("png"))
    except Exception:
        pass


def _parse_table_rows(page: fitz.Page, clip: fitz.Rect) -> list[list[str]]:
    """从裁剪区域内解析表格行（按行 y、列 x 聚类）。"""
    raw_lines: list[tuple[float, list[tuple[float, str]]]] = []

    for block in page.get_text("dict", clip=clip).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            y = line["bbox"][1]
            spans = [(s["bbox"][0], s.get("text", "").strip()) for s in line.get("spans", [])]
            spans = [(x, t) for x, t in spans if t]
            if spans:
                raw_lines.append((y, spans))

    if not raw_lines:
        return []

    raw_lines.sort(key=lambda item: item[0])
    row_groups: list[list[tuple[float, str]]] = []
    current_y = raw_lines[0][0]
    current_spans: list[tuple[float, str]] = []

    for y, spans in raw_lines:
        if abs(y - current_y) <= 4:
            current_spans.extend(spans)
        else:
            if current_spans:
                row_groups.append(current_spans)
            current_spans = list(spans)
            current_y = y
    if current_spans:
        row_groups.append(current_spans)

    col_bounds = _detect_column_bounds(row_groups)
    if len(col_bounds) < 2:
        return _simple_rows(row_groups)

    rows: list[list[str]] = []
    for spans in row_groups:
        cells = _spans_to_cells(spans, col_bounds)
        cells = [c.strip() for c in cells if c.strip()]
        if len(cells) >= 2:
            rows.append(cells)

    return _merge_method_cells(rows)


def _detect_column_bounds(row_groups: list[list[tuple[float, str]]]) -> list[float]:
    """从首行或多行推断列边界 x 坐标。"""
    xs: list[float] = []
    for spans in row_groups[:6]:
        for x, _ in spans:
            xs.append(x)
    if not xs:
        return []

    xs.sort()
    bounds: list[float] = [xs[0] - 10]
    for x in xs[1:]:
        if x - bounds[-1] > 35:
            bounds.append(x - 5)
    bounds.append(xs[-1] + 200)
    return bounds


def _spans_to_cells(spans: list[tuple[float, str]], col_bounds: list[float]) -> list[str]:
    cells = [""] * (len(col_bounds) - 1)
    for x, text in sorted(spans, key=lambda item: item[0]):
        col = 0
        for i in range(len(col_bounds) - 1):
            if col_bounds[i] <= x < col_bounds[i + 1]:
                col = i
                break
        cells[col] = (cells[col] + " " + text).strip() if cells[col] else text
    return cells


def _simple_rows(row_groups: list[list[tuple[float, str]]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for spans in row_groups:
        cells = [t for _, t in sorted(spans, key=lambda item: item[0])]
        if len(cells) >= 2:
            rows.append(cells)
    return rows


def _clean_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """合并 ± 拆分单元格，过滤明显非表格行。"""
    if not rows:
        return []

    cleaned: list[list[str]] = []
    for row in rows:
        merged: list[str] = []
        i = 0
        while i < len(row):
            cell = row[i].strip()
            if cell in ("±", "±") and merged:
                merged[-1] = f"{merged[-1]} ± {row[i + 1].strip()}".strip()
                i += 2
                continue
            if cell == "±" and i + 1 < len(row):
                if merged:
                    merged[-1] = f"{merged[-1]} ± {row[i + 1].strip()}".strip()
                i += 2
                continue
            merged.append(cell)
            i += 1
        text = " ".join(merged)
        if len(text) > 140 and not TABLE_HEADER_HINTS.search(text):
            continue
        if len(merged) >= 2:
            cleaned.append(merged)
    return cleaned


def rows_look_valid(rows: list[list[str]]) -> bool:
    """判断解析结果是否适合渲染为 PPT 原生表格。"""
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    header = " ".join(rows[0]).lower()
    if not TABLE_HEADER_HINTS.search(header):
        return False
    numeric_rows = sum(
        1 for row in rows[1:]
        if any(NUMERIC_CELL.search(c) for c in row)
    )
    return numeric_rows >= 1


def _merge_method_cells(rows: list[list[str]]) -> list[list[str]]:
    """合并被拆散的方法名（如 'EWC [Kirkpatrick' + 'et al.'）。"""
    merged: list[list[str]] = []
    for row in rows:
        if merged and len(row) > len(merged[-1]) and row[0] and not NUMERIC_CELL.search(row[0]):
            prev = merged[-1]
            if len(prev) == len(row) - 1:
                prev[0] = f"{prev[0]} {row[0]}".strip()
                prev.extend(row[1:])
                continue
        merged.append(row)
    return merged
