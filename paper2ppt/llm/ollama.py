"""
Ollama 本地大模型后端 — 通过 HTTP 调用 Ollama 服务。

默认服务地址: http://localhost:11434
官方 API 文档: https://github.com/ollama/ollama/blob/main/docs/api.md

用到的接口:
  GET  /api/tags  — 列出本机已 pull 的模型
  POST /api/chat  — 对话补全（stream=false 时一次性返回完整回复）

自动选模逻辑:
  若未指定 --model，调用 list_models() 获取全部模型名，
  用 _parse_model_size() 从名称中解析参数量（如 qwen3:4b → 4.0），
  选择参数量最大的模型。
"""

from __future__ import annotations

import re

import requests

from paper2ppt.llm.base import LLMBackend

# Ollama 默认监听地址，可通过环境变量或后续扩展参数修改
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _parse_model_size(name: str) -> float:
    """
    从 Ollama 模型标签中估算参数量（单位：B，十亿）。

    匹配模式: 数字 + 可选小数 + 'b'，如 4b、7b、70b、1.5b。
    无法解析时返回 0.0，在 max() 比较中优先级最低。

    示例:
        "qwen3:4b"    → 4.0
        "llama3:70b"  → 70.0
        "nomic-embed" → 0.0（嵌入模型无 b 后缀）
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", name.lower())
    if match:
        return float(match.group(1))
    return 0.0


class OllamaBackend(LLMBackend):
    """
    Ollama 本地服务后端。

    属性:
        base_url: Ollama API 根地址（无尾部斜杠）
        model   : 当前使用的模型名，如 "qwen3:4b"
    """

    def __init__(self, model: str | None = None, base_url: str = DEFAULT_OLLAMA_URL):
        self.base_url = base_url.rstrip("/")
        # 未指定 model 时自动选择本机最大模型
        self.model = model or self._auto_select_model()

    def _auto_select_model(self) -> str:
        """
        自动选择参数量最大的已安装模型。

        若无任何模型，抛出 RuntimeError 并提示用户 ollama pull。
        """
        models = self.list_models()
        if not models:
            raise RuntimeError(
                "未检测到已安装的 Ollama 模型。请先运行 `ollama pull <model>` 下载模型，"
                "或通过 --model 参数指定模型名称。"
            )
        return max(models, key=_parse_model_size)

    def list_models(self) -> list[str]:
        """
        调用 GET /api/tags 获取已安装模型列表。

        连接失败时抛出 RuntimeError，提示安装/启动 Ollama。
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise RuntimeError(
                "无法连接到 Ollama 服务。请确认已安装并启动 Ollama："
                "https://ollama.com/"
            ) from None
        data = resp.json()
        # 响应格式: {"models": [{"name": "qwen3:4b", ...}, ...]}
        return [m["name"] for m in data.get("models", [])]

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.3) -> str:
        """
        调用 POST /api/chat 进行对话补全。

        stream=false 表示等待完整回复后一次性返回（便于 JSON 解析）。
        timeout=600 秒，长论文分析可能较慢，可按需调整。

        响应格式: {"message": {"role": "assistant", "content": "..."}, ...}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
