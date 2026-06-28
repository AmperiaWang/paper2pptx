"""
LLM 子包 — 所有大模型调用均通过 HTTP 网络接口实现。

对外导出:
  LLMBackend    — 抽象基类，定义 chat / list_models 接口
  create_backend — 工厂函数，按名称创建 ollama / openai / deepseek 实例

模块结构:
  base.py              — 抽象接口
  ollama.py            — 本地 Ollama 服务 (localhost:11434)
  openai_compatible.py — OpenAI / DeepSeek 等兼容 API
  factory.py           — 后端创建与注册
"""

from paper2ppt.llm.base import LLMBackend
from paper2ppt.llm.factory import create_backend

__all__ = ["LLMBackend", "create_backend"]
