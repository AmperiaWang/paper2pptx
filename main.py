#!/usr/bin/env python3
"""Paper2PPT 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper2ppt.llm.factory import SUPPORTED_BACKENDS, create_backend
from paper2ppt.pipeline import DEFAULT_TEMPLATE, run_pipeline
from paper2ppt.prompts import DEFAULT_LANG, LANG_ALIASES, SUPPORTED_LANGS, normalize_lang


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将研究论文 PDF 转换为 PowerPoint 演示文稿",
    )
    parser.add_argument("pdf", help="输入的论文 PDF 文件路径")
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=SUPPORTED_BACKENDS,
        help="LLM 后端 (默认: ollama)",
    )
    parser.add_argument("--apikey", help="在线大模型 API Key (openai/deepseek 必填)")
    parser.add_argument("--model", help="指定模型名称")
    parser.add_argument(
        "--output", "-o",
        help="输出 PPT 文件路径 (默认: 与 PDF 同名的 .pptx)",
    )
    parser.add_argument(
        "--prompt",
        help="自定义提示词文件路径 (默认: prompt.json)",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        choices=[*SUPPORTED_LANGS, *LANG_ALIASES.keys()],
        help="PPT 语言 (默认: zh_cn 简体中文; en_us 美式英文)",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="PPT 母版模板文件 (默认: template/index.pptx)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"错误: 找不到文件 {pdf_path}", file=sys.stderr)
        return 1

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"错误: 找不到模板文件 {template_path}", file=sys.stderr)
        return 1

    try:
        backend = create_backend(
            args.backend,
            api_key=args.apikey,
            model=args.model,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.backend == "ollama":
        print(f"使用 Ollama 模型: {backend.model}")

    try:
        run_pipeline(
            pdf_path,
            backend,
            output_path=args.output,
            prompt_path=args.prompt,
            lang=normalize_lang(args.lang),
            template_path=template_path,
        )
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
