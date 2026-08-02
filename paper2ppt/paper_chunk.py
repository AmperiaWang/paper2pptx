"""
论文分片阅读 — 将长论文切分为多段，逐段摘要后再合成幻灯片。

Map-Reduce 流程:
  1. split_text_chunks — 按段落边界切分原文
  2. summarize_chunks  — 每段调用 LLM 提取结构化笔记（Map）
  3. merge_chunk_notes   — 合并笔记供最终 analyze 使用（Reduce 输入）
"""

from __future__ import annotations

import re

from tqdm import tqdm

from paper2ppt.llm.base import LLMBackend
from paper2ppt.llm.ollama import OllamaBackend

# 超过此长度启用分片；单段上限字符数
CHUNK_THRESHOLD = 20_000
CHUNK_SIZE = 12_000

# 分片摘要 / 最终合成 的 Ollama num_ctx（较小上下文推理更快）
CHUNK_NUM_CTX = 8_192
SYNTHESIS_NUM_CTX = 24_576

CHUNK_TIMEOUT = 180
SYNTHESIS_TIMEOUT = 600


def split_text_chunks(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """按段落边界切分文本，避免在句中硬切。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sep_len = 2 if current else 0
        if current_len + sep_len + len(para) <= max_chars:
            current.append(para)
            current_len += sep_len + len(para)
            continue

        if current:
            chunks.append("\n\n".join(current))

        if len(para) > max_chars:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            current = []
            current_len = 0
        else:
            current = [para]
            current_len = len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def merge_chunk_notes(notes: list[str]) -> str:
    """将各段摘要合并为一份结构化笔记。"""
    parts: list[str] = []
    for i, note in enumerate(notes, start=1):
        note = note.strip()
        if note:
            parts.append(f"### 片段 {i}\n{note}")
    header = (
        "以下是从论文分片阅读得到的结构化笔记（按原文顺序）。"
        "请基于这些笔记撰写演示文稿，勿臆造笔记中未出现的内容。\n\n"
    )
    return header + "\n\n".join(parts)


def _backend_chat(
    backend: LLMBackend,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    num_ctx: int | None = None,
    timeout: int = 600,
) -> str:
    """调用 backend.chat，Ollama 时支持 num_ctx / timeout 以加速分片请求。"""
    if isinstance(backend, OllamaBackend):
        return backend.chat(
            messages,
            temperature=temperature,
            num_ctx=num_ctx,
            timeout=timeout,
        )
    return backend.chat(messages, temperature=temperature)


def summarize_chunks(
    backend: LLMBackend,
    paper_text: str,
    prompts: dict[str, str],
    *,
    chunk_size: int = CHUNK_SIZE,
) -> str:
    """
    分片摘要论文，返回合并后的结构化笔记。

    打印进度信息，便于长论文处理时感知进展。
    """
    chunks = split_text_chunks(paper_text, chunk_size)
    total = len(chunks)
    template = prompts.get("summarize_chunk")
    if not template:
        raise ValueError("prompt.json is missing summarize_chunk prompt")

    notes: list[str] = []
    system = prompts["system"]

    with tqdm(chunks, desc="      Chunk summary", unit="chunk", dynamic_ncols=True) as bar:
        for index, chunk in enumerate(bar, start=1):
            bar.set_postfix_str(f"{len(chunk)} chars", refresh=False)
            user_prompt = template.format(
                index=index,
                total=total,
                chunk_text=chunk,
            )
            user_prompt += (
                "\n\n请保留可用于制作幻灯片的细节：每个模块的输入输出、"
                "步骤与设计原因；数据集、baseline、指标和所有关键数字；"
                "Figure/Table 编号及其支撑的结论。不要只写一句概括。"
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
            note = _backend_chat(
                backend,
                messages,
                temperature=0.2,
                num_ctx=CHUNK_NUM_CTX,
                timeout=CHUNK_TIMEOUT,
            )
            notes.append(note.strip())

    merged = merge_chunk_notes(notes)
    print(f"      Merged {total} chunk summaries ({len(merged)} chars)")
    return merged


def should_chunk(paper_text: str, threshold: int = CHUNK_THRESHOLD) -> bool:
    """判断是否需要启用分片分析。"""
    return len(paper_text) > threshold


def build_narrative(
    backend: LLMBackend,
    paper_text: str,
    prompts: dict[str, str],
    paper_structure: str = "",
) -> str:
    """生成贯穿全文的逻辑推理链；paper_structure 约束各节 focus 不遗漏。"""
    template = prompts.get("build_narrative")
    if not template:
        return ""

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": template.format(
                paper_text=paper_text[:MAX_NARRATIVE_CHARS],
                paper_structure=paper_structure or "(default outline)",
            ),
        },
    ]
    with tqdm(total=1, desc="      Narrative outline", bar_format="{desc}: {elapsed}") as bar:
        result = _backend_chat(
            backend,
            messages,
            temperature=0.25,
            num_ctx=SYNTHESIS_NUM_CTX,
            timeout=SYNTHESIS_TIMEOUT,
        ).strip()
        bar.update(1)
    return result


MAX_NARRATIVE_CHARS = 50_000
