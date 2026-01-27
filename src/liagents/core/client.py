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
        **kwargs,
    ) -> Iterator[str]:
        """
        基础流式调用

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            流式响应迭代器
        """
        print(f"正在调用 {self._model} 模型...")
        try:
            response = self._client.chat.completions.create(
                messages=messages,
                model=kwargs.pop("model", self._model),
                temperature=kwargs.pop("temperature", self._temperature),
                max_completion_tokens=kwargs.pop(
                    "max_completion_tokens", self._max_completion_tokens
                ),
                stream=True,
                **kwargs,
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

    def stream_chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Union[str, dict] = "auto",
        **kwargs,
    ) -> Iterator[str]:
        """
        支持工具调用的流式方法

        Args:
            messages: 消息列表
            tools: 工具schemas（OpenAI函数调用格式）
            tool_choice: 工具选择策略 ("auto", "none", 或具体工具)
            **kwargs: 其他参数

        Returns:
            流式响应迭代器
        """
        print(f"正在调用 {self._model} 模型...")
        try:
            create_params = {
                "messages": messages,
                "model": kwargs.pop("model", self._model),
                "temperature": kwargs.pop("temperature", self._temperature),
                "max_completion_tokens": kwargs.pop(
                    "max_completion_tokens", self._max_completion_tokens
                ),
                "stream": True,
                "tools": tools,
                "tool_choice": tool_choice,
                **kwargs,
            }

            response = self._client.chat.completions.create(**create_params)

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

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Union[str, dict] = "auto",
        **kwargs,
    ) -> Any:
        """
        支持函数调用的聊天方法，返回完整响应对象

        Args:
            messages: 消息列表
            tools: 工具schemas（OpenAI函数调用格式）
            tool_choice: 工具选择策略 ("auto", "none", 或具体工具)
            **kwargs: 其他参数

        Returns:
            完整的OpenAI响应对象
        """
        try:
            response = self._client.chat.completions.create(
                messages=messages,
                model=kwargs.pop("model", self._model),
                temperature=kwargs.pop("temperature", self._temperature),
                max_completion_tokens=kwargs.pop(
                    "max_completion_tokens", self._max_completion_tokens
                ),
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
                **kwargs,
            )
            return response
        except Exception as e:
            raise RuntimeError(f"调用LLM API时发生错误: {e}")
