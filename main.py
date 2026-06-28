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
        description="Convert a research paper PDF into a PowerPoint presentation",
    )
    parser.add_argument("pdf", help="Input PDF file path")
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=SUPPORTED_BACKENDS,
        help="LLM backend (default: ollama)",
    )
    parser.add_argument("--apikey", help="API key for online LLMs (required for openai/deepseek)")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument(
        "--output", "-o",
        help="Output .pptx path (default: same name as PDF)",
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt file (default: prompt.json)",
    )
    parser.add_argument(
        "--structure",
        default=None,
        help="Deck structure JSON (default: paper_structure.json)",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        choices=[*SUPPORTED_LANGS, *LANG_ALIASES.keys()],
        help="Slide language (default: zh_cn; en_us for English)",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="PowerPoint template file (default: template/index.pptx)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        return 1

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"Error: template not found: {template_path}", file=sys.stderr)
        return 1

    try:
        backend = create_backend(
            args.backend,
            api_key=args.apikey,
            model=args.model,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.backend == "ollama":
        print(f"Using Ollama model: {backend.model}")

    try:
        run_pipeline(
            pdf_path,
            backend,
            output_path=args.output,
            prompt_path=args.prompt,
            structure_path=args.structure,
            lang=normalize_lang(args.lang),
            template_path=template_path,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
