"""
文本 inline 格式解析 — 将 LLM 输出的 Markdown 风格标记转为 PPT run 属性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# LLM 元话语 / 过渡套话（不应出现在幻灯片上）
META_PHRASE_RE = re.compile(
    r"承接|主线|本页|我们先|进一步探讨|基于报告|方法链条|让我|下面来看|"
    r"接下来|如上所述|正如前面|回到|梳理|厘清|验证其在|是否成立",
    re.IGNORECASE,
)

# **bold** / *italic* （非贪婪）
_INLINE_RE = re.compile(r"\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*")


@dataclass
class TextRun:
    text: str
    bold: bool = False
    italic: bool = False


def parse_inline_runs(text: str) -> list[TextRun]:
    """解析 **粗体**、*斜体*、***粗斜体***。"""
    runs: list[TextRun] = []
    pos = 0
    for match in _INLINE_RE.finditer(text):
        if match.start() > pos:
            runs.append(TextRun(text[pos : match.start()]))
        if match.group(1):
            runs.append(TextRun(match.group(1), bold=True, italic=True))
        elif match.group(2):
            runs.append(TextRun(match.group(2), bold=True))
        elif match.group(3):
            runs.append(TextRun(match.group(3), italic=True))
        pos = match.end()
    if pos < len(text):
        runs.append(TextRun(text[pos:]))
    if not runs:
        runs.append(TextRun(text))
    return [r for r in runs if r.text]


def strip_markdown(text: str) -> str:
    """去掉格式标记，保留纯文本。"""
    plain = text
    plain = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", plain)
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
    plain = re.sub(r"\*(.+?)\*", r"\1", plain)
    return plain


def is_meta_sentence(text: str) -> bool:
    """判断是否为元话语/AI 腔过渡句。"""
    t = strip_markdown(text).strip()
    if not t:
        return True
    if META_PHRASE_RE.search(t):
        return True
    return t.startswith(("▸", "→", "=>"))
