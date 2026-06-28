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

import json
import re

import requests

from paper2ppt.llm.base import LLMBackend

# Ollama 默认监听地址，可通过环境变量或后续扩展参数修改
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Ollama 未显式设置 num_ctx 时默认仅 4096，不足以分析整篇论文
DEFAULT_NUM_CTX = 32_768


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
        self.num_ctx = self._resolve_num_ctx(self.model)

    def _auto_select_model(self) -> str:
        """
        自动选择参数量最大的已安装模型。

        若无任何模型，抛出 RuntimeError 并提示用户 ollama pull。
        """
        models = self.list_models()
        if not models:
            raise RuntimeError(
                "No Ollama models found. Run `ollama pull <model>` first, "
                "or pass --model to specify a model."
            )
        return max(models, key=_parse_model_size)

    def _fetch_tags(self) -> list[dict]:
        """调用 GET /api/tags，返回 models 数组。"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure it is installed and running: "
                "https://ollama.com/"
            ) from None
        return resp.json().get("models", [])

    def _resolve_num_ctx(self, model_name: str) -> int:
        """
        从 /api/tags 读取模型 context_length，作为 chat 请求的 num_ctx。

        Ollama 默认 num_ctx 仅 4096，整篇论文会触发 400 exceed_context_size。
        """
        for entry in self._fetch_tags():
            if entry.get("name") == model_name or entry.get("model") == model_name:
                ctx = entry.get("details", {}).get("context_length")
                if ctx:
                    return int(ctx)
        return DEFAULT_NUM_CTX

    def list_models(self) -> list[str]:
        """
        调用 GET /api/tags 获取已安装模型列表。

        连接失败时抛出 RuntimeError，提示安装/启动 Ollama。
        """
        return [m["name"] for m in self._fetch_tags()]

    @staticmethod
    def _format_api_error(resp: requests.Response) -> str:
        """解析 Ollama 错误响应，返回可读说明。"""
        try:
            body = resp.json()
        except ValueError:
            return resp.text or f"HTTP {resp.status_code}"

        err = body.get("error", "")
        if isinstance(err, str):
            if err.startswith("{"):
                try:
                    inner = json.loads(err)
                    if isinstance(inner, dict):
                        nested = inner.get("error", inner)
                        if isinstance(nested, dict) and nested.get("message"):
                            return nested["message"]
                except json.JSONDecodeError:
                    pass
            if err:
                return err
        return resp.text or f"HTTP {resp.status_code}"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        num_ctx: int | None = None,
        timeout: int = 600,
    ) -> str:
        """
        调用 POST /api/chat 进行对话补全。

        stream=false 表示等待完整回复后一次性返回（便于 JSON 解析）。
        num_ctx 可逐请求覆盖，分片摘要时使用较小上下文以加速推理。

        响应格式: {"message": {"role": "assistant", "content": "..."}, ...}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx if num_ctx is not None else self.num_ctx,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
        except requests.ReadTimeout:
            raise RuntimeError(
                f"Ollama timed out ({timeout}s). "
                "Long papers are chunked automatically; if it still times out, "
                "try a smaller model: --model qwen2.5-coder:latest"
            ) from None
        if not resp.ok:
            detail = self._format_api_error(resp)
            raise RuntimeError(
                f"Ollama request failed ({resp.status_code}): {detail}"
            ) from None
        data = resp.json()
        message = data.get("message", {})
        content = (message.get("content") or "").strip()
        if content:
            return content
        # thinking 模型可能把正文放在 thinking 字段
        thinking = (message.get("thinking") or "").strip()
        if thinking:
            return thinking
        return content
