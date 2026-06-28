"""
幻灯片后处理 — 去掉元话语、合并无用 bridge、优化标题。
"""

from __future__ import annotations

import copy
import re

from paper2ppt.text_format import is_meta_sentence, strip_markdown

_ROADMAP_TITLES = frozenset({"报告主线", "论文脉络", "talk roadmap", "roadmap"})


def polish_slides(slides: list[dict], lang: str = "zh_cn") -> list[dict]:
    """清理 LLM 输出中不宜出现在 PPT 上的字段与措辞。"""
    result: list[dict] = []
    for slide in slides:
        s = copy.deepcopy(slide)
        if s.get("type") != "content":
            result.append(s)
            continue

        s.pop("bridge", None)

        title = (s.get("title") or "").strip()
        if not title:
            s["title"] = s.get("section") or ("要点" if lang.startswith("zh") else "Key Points")
            title = s["title"]
        if title.lower() in _ROADMAP_TITLES or title in _ROADMAP_TITLES:
            s["title"] = "全文概览" if lang.startswith("zh") else "Talk Overview"

        bullets = []
        for raw in s.get("bullets") or []:
            text = str(raw).strip()
            if not text:
                continue
            plain = strip_markdown(text)
            if is_meta_sentence(plain):
                continue
            bullets.append(text)

        s["bullets"] = bullets
        result.append(s)
    return result
