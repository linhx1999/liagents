import os
from openai import OpenAI
from typing import List, Dict, Optional, Iterator, Union, Any


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
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Union[str, dict] = "auto",
        **kwargs,
    ) -> Iterator[str]:
        """
        流式调用，支持工具调用

        Args:
            messages: 消息列表
            temperature: 温度参数
            tools: 工具schemas（OpenAI函数调用格式）
            tool_choice: 工具选择策略
            **kwargs: 其他参数

        Returns:
            流式响应迭代器
        """
        print(f"正在调用 {self._model} 模型...")
        try:
            request_params = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature or self._temperature,
                "max_completion_tokens": self._max_completion_tokens,
                "stream": True,
            }

            # 添加工具调用支持
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = tool_choice

            # 添加其他参数
            request_params.update(kwargs)

            response = self._client.chat.completions.create(**request_params)

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

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                messages=messages,
                model=kwargs.pop("model", self._model),
                temperature=kwargs.pop("temperature", self._temperature),
                max_completion_tokens=kwargs.pop(
                    "max_completion_tokens", self._max_completion_tokens
                ),
                stream=False,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"调用LLM API时发生错误: {e}")
