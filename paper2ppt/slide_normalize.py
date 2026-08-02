"""
幻灯片结构规范化 — 将 section 章节标题合并到内容页，避免单独占一页。
"""

from __future__ import annotations

import copy


def flatten_sections(slides: list[dict]) -> list[dict]:
    """
    把 type=section 的章节标题（如「引言」「实验」）合并到紧随其后的 content 页：

    - 不再生成单独的 section 幻灯片
    - 该章节第一页 content 的 title 使用章节名（如「引言」）
    - 原 content 子标题（如「研究背景」）并入 bullets 首条
    - 在 content 上附加 section 字段，供 Figure 匹配等逻辑使用
    """
    result: list[dict] = []
    current_section = ""
    first_in_section = True

    for slide in slides:
        slide_type = slide.get("type", "content")

        if slide_type == "section":
            current_section = (slide.get("title") or "").strip()
            first_in_section = True
            result.append(copy.deepcopy(slide))
            continue

        if slide_type == "title":
            current_section = ""
            first_in_section = True
            result.append(copy.deepcopy(slide))
            continue

        if slide_type != "content":
            result.append(copy.deepcopy(slide))
            continue

        new_slide = copy.deepcopy(slide)
        if current_section:
            new_slide["section"] = current_section

        # 保留 LLM 给出的具体页标题（如「动态结构调控机制」），不再用章节名覆盖
        if current_section and first_in_section and not (new_slide.get("title") or "").strip():
            new_slide["title"] = current_section

        first_in_section = False
        result.append(new_slide)

    return result
