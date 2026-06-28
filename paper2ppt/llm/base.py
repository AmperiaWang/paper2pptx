"""
LLM 后端抽象基类。

所有后端（Ollama、OpenAI、DeepSeek）均通过 HTTP 网络接口通信，
本模块定义统一接口，便于 pipeline 与具体实现解耦。

新增后端步骤：
  1. 继承 LLMBackend，实现 chat() 和 list_models()
  2. 在 factory.create_backend() 中注册
  3. 将新 backend 名称加入 SUPPORTED_BACKENDS
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """
    LLM 后端的抽象基类。

    子类必须实现 chat 和 list_models，均通过网络 HTTP 请求完成。
  """

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.3) -> str:
        """
        发送对话补全请求，返回助手回复文本。

        参数:
            messages   : OpenAI 格式的消息列表，每项含 role 和 content。
                         role 通常为 "system" | "user" | "assistant"。
            temperature: 采样温度，越低输出越确定（默认 0.3，适合结构化 JSON）。

        返回:
            助手回复的纯文本字符串（pipeline 会再用 extract_json 解析）。
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        查询当前后端可用的模型名称列表。

        用于 Ollama 自动选模；在线 API 失败时可回退到当前配置的 model。
        """
