import os
from openai import OpenAI
from typing import List, Dict, Optional, Iterator


class Client:
    """模型客户端类，提供统一的模型调用接口"""

    # 实例属性类型声明
    _model: str
    _client: OpenAI
    _temperature: float
    _max_completion_tokens: Optional[int] = None

    def __init__(
        self,
        model: str = os.getenv("MODEL", ""),
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        base_url: str = os.getenv("OPENAI_BASE_URL", ""),
        temperature: float = 0.7,
        max_completion_tokens: Optional[int] = None,
        timeout: int = 60,
    ):
        self._model = model
        self._temperature = temperature
        self._max_completion_tokens = max_completion_tokens

        if not all([self._model, api_key, base_url]):
            raise ValueError("必须提供MODEL、OPENAI_API_KEY和OPENAI_BASE_URL")

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def stream_chat(
        self, messages: List[Dict[str, str]], temperature: Optional[float] = None
    ) -> Iterator[str]:
        print(f"正在调用 {self._model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature or self._temperature,
                max_completion_tokens=self._max_completion_tokens,
                stream=True,
            )

            # 处理流式响应
            print("大语言模型响应成功:")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            raise RuntimeError(f"调用LLM API时发生错误: {e}")

    def invoke_chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式调用，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", self._temperature),
                max_completion_tokens=kwargs.get(
                    "max_completion_tokens", self._max_completion_tokens
                ),
                stream=False,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["temperature", "max_completion_tokens", "stream"]
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"调用LLM API时发生错误: {e}")
