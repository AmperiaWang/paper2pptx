"""
LLM 后端工厂 — 根据 CLI 参数创建对应的后端实例。

这是 backend 选择的唯一入口，main.py 通过 create_backend() 获取 LLM 客户端，
无需直接 import 各具体实现类。

支持的 backend 名称见 SUPPORTED_BACKENDS；
新增后端时在此注册映射关系即可。
"""

from __future__ import annotations

from paper2ppt.llm.base import LLMBackend
from paper2ppt.llm.ollama import OllamaBackend
from paper2ppt.llm.openai_compatible import (
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    OpenAICompatibleBackend,
)

# CLI --backend 的可选值，与 create_backend 分支一一对应
SUPPORTED_BACKENDS = ("ollama", "openai", "deepseek")


def create_backend(
    backend: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMBackend:
    """
    根据名称创建 LLM 后端实例。

    参数:
        backend : 后端名称，不区分大小写。
        api_key : 在线 API 密钥；ollama 不需要。
        model   : 模型名称；为 None 时各后端使用自己的默认策略。

    返回:
        实现了 LLMBackend 接口的实例。

    抛出:
        ValueError   — 不支持的 backend 名称，或在线 backend 缺少 apikey。
        RuntimeError — ollama 服务不可用（由 OllamaBackend 抛出）。
    """
    backend = backend.lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. Choose: {', '.join(SUPPORTED_BACKENDS)}"
        )

    if backend == "ollama":
        # Ollama 走本地 HTTP，无需 api_key
        return OllamaBackend(model=model)

    # openai 与 deepseek 共用 OpenAICompatibleBackend，仅 base_url 不同
    base_urls = {
        "openai": OPENAI_BASE_URL,
        "deepseek": DEEPSEEK_BASE_URL,
    }
    return OpenAICompatibleBackend(
        api_key=api_key or "",
        model=model,
        base_url=base_urls[backend],
        backend_name=backend,
    )
