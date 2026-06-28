"""
提示词加载与 LLM 输出解析。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SUPPORTED_LANGS = ("zh_cn", "en_us")
DEFAULT_LANG = "zh_cn"

# 兼容旧参数名
LANG_ALIASES = {
    "zh": "zh_cn",
    "chinese": "zh_cn",
    "en": "en_us",
    "english": "en_us",
}


def normalize_lang(lang: str) -> str:
    """统一语言参数。"""
    key = lang.lower().replace("-", "_")
    return LANG_ALIASES.get(key, key)


def load_prompts(
    path: str | Path | None = None,
    lang: str = DEFAULT_LANG,
) -> dict[str, str]:
    lang = normalize_lang(lang)
    if lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"不支持的语言: {lang}。可选: {', '.join(SUPPORTED_LANGS)}"
        )

    prompt_path = Path(path) if path else Path(__file__).resolve().parent.parent / "prompt.json"
    with open(prompt_path, encoding="utf-8") as f:
        raw = json.load(f)

    if lang in raw and isinstance(raw[lang], dict):
        return raw[lang]

    return {
        "system": raw.get("system", ""),
        "analyze_paper": raw.get("analyze_paper", ""),
    }


def extract_json(text: str) -> dict:
    """从 LLM 回复中提取 JSON 对象。"""
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    return json.loads(text)
