"""
OpenAI 兼容 API 后端 — 适用于 OpenAI、DeepSeek 等在线大模型。

两者均实现 OpenAI Chat Completions 协议，仅 base_url 和默认模型不同：
  OpenAI  : https://api.openai.com/v1
  DeepSeek: https://api.deepseek.com/v1

用到的接口:
  GET  /models            — 列出可用模型（可选，失败时回退到当前 model）
  POST /chat/completions  — 对话补全

认证方式: HTTP Header `Authorization: Bearer <api_key>`

扩展其他兼容服务（如 Azure OpenAI、本地 vLLM）:
  传入对应 base_url 即可，无需改 chat 逻辑。
"""

from __future__ import annotations

import requests

from paper2ppt.llm.base import LLMBackend

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 各 backend 未指定 --model 时使用的默认模型名
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
}


class OpenAICompatibleBackend(LLMBackend):
    """
    OpenAI Chat Completions 兼容后端。

    属性:
        api_key      : API 密钥
        base_url     : API 根地址（含 /v1）
        model        : 模型 ID，如 gpt-4o、deepseek-chat
        backend_name : 逻辑名称，用于错误提示（openai / deepseek）
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = OPENAI_BASE_URL,
        backend_name: str = "openai",
    ):
        if not api_key:
            raise ValueError(f"--apikey is required for the {backend_name} backend.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model or DEFAULT_MODELS.get(backend_name, "gpt-4o")
        self.backend_name = backend_name

    def list_models(self) -> list[str]:
        """
        调用 GET /models 获取账号下可用模型 ID 列表。

        部分服务商不支持此接口或需要特殊权限，失败时返回 [self.model] 作为兜底。
        """
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI 格式: {"data": [{"id": "gpt-4o", ...}, ...]}
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return [self.model]

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.3) -> str:
        """
        调用 POST /chat/completions 进行对话补全。

        非 2xx 响应时抛出 RuntimeError，错误信息含 status_code 和响应体，
        便于排查鉴权失败、模型不存在、余额不足等问题。

        响应格式: {"choices": [{"message": {"content": "..."}}], ...}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=600,
        )
        if not resp.ok:
            raise RuntimeError(
                f"{self.backend_name} API request failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()["choices"][0]["message"]["content"]

    def _headers(self) -> dict[str, str]:
        """构造 HTTP 请求头，含 Bearer 认证。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
