import os
from openai import OpenAI
from typing import List, Dict, Optional, Iterator, Union, Any
from collections import defaultdict

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
        ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
            # 1. 参数构建
            create_params = {
                "messages": messages,
                "model": kwargs.pop("model", self._model),
                "temperature": kwargs.pop("temperature", self._temperature),
                "max_completion_tokens": kwargs.pop("max_completion_tokens", self._max_completion_tokens),
                "stream": True,
                **kwargs,
            }
            
            # 只有当 tools 存在时才传入相关参数，避免 API 报错
            if tools:
                create_params["tools"] = tools
                create_params["tool_choice"] = tool_choice

            try:
                print(f"正在调用 {create_params['model']} 模型...")
                response = self._client.chat.completions.create(**create_params)
                
                # 2. 核心状态容器
                # 我们必须返回一个确定的列表对象，以便在生成器运行时修改它
                # 注意：这里不能默认返回 None，因为流还没开始跑，我们不知道有没有工具
                final_tool_calls: list[dict[str, Any]] = []
                
                # 内部缓冲区，用于按 index 组装碎片
                tool_buffer = defaultdict(lambda: {"id": None, "function": {"name": "", "arguments": ""}, "type": "function"})

                def stream_generator():
                    print("大语言模型响应成功，开始流式传输...")
                    
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        
                        # --- A. 处理文本内容 ---
                        if delta.content:
                            print(delta.content, end="", flush=True)
                            yield delta.content

                        # --- B. 处理工具调用 ---
                        if delta.tool_calls:
                            for tool_chunk in delta.tool_calls:
                                idx = tool_chunk.index
                                
                                # 1. ID 和 Name 通常只在首个 chunk 出现
                                if tool_chunk.id:
                                    tool_buffer[idx]["id"] = tool_chunk.id
                                if tool_chunk.function and tool_chunk.function.name:
                                    tool_buffer[idx]["function"]["name"] = tool_chunk.function.name
                                
                                # 2. Arguments 会分散在多个 chunk 中，需要拼接
                                if tool_chunk.function and tool_chunk.function.arguments:
                                    tool_buffer[idx]["function"]["arguments"] += tool_chunk.function.arguments
                    
                    # --- C. 流结束后的处理 ---
                    # 将 buffer 中的数据整理到 final_tool_calls 列表中
                    # 按照 index 排序确保顺序正确
                    if tool_buffer:
                        sorted_indices = sorted(tool_buffer.keys())
                        for idx in sorted_indices:
                            tool_data = tool_buffer[idx]
                            # 这里构造成标准的 tool_call 结构
                            final_tool_calls.append({
                                "id": tool_data["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_data["function"]["name"],
                                    "arguments": tool_data["function"]["arguments"]
                                }
                            })
                        
                # 返回生成器和列表引用
                return stream_generator(), final_tool_calls

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
