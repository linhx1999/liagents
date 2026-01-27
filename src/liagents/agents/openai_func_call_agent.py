import json
from typing import Iterator, Optional, Union, Any, Dict

from ..core.agent import Agent
from ..core.config import Config
from ..core.client import Client
from ..core.message import Message
from ..tools.registry import ToolRegistry


def _map_parameter_type(param_type: str) -> str:
    """将工具参数类型映射为JSON Schema允许的类型"""
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


class OpenAIFuncCallAgent(Agent):
    """基于OpenAI原生函数调用机制的Agent"""

    def __init__(
        self,
        name: str,
        client: Client,
        system_prompt: str = "",
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 10,
    ):
        super().__init__(name, client, system_prompt.strip(), config)
        self.tool_registry = tool_registry or ToolRegistry()
        self._history: list[Message] = []
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations

    def _build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """构建初始消息列表"""
        messages: list[dict[str, Any]] = []
        messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        # Tool对象
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: list[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        return schemas


    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """解析模型返回的JSON字符串参数"""
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """根据工具定义尽可能转换参数类型"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用并返回字符串结果"""
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                user_input = arguments.get("input", "")
                return func(user_input)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        return f"❌ 错误：未找到工具 '{tool_name}'"

    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("Client 未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    def run(
        self,
        user_input: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        """
        执行函数调用范式的对话流程
        """
        messages = self._build_messages(user_input)
        tool_schemas = self._build_tool_schemas()
        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice
        current_iteration = 0
        final_response = ""

        while current_iteration < iterations_limit:
            response = self.client.invoke_chat(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            assistant_message = response.choices[0].message or {"role": "assistant", "content": ""}
            tool_calls = list(assistant_message.tool_calls or [])

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                result = self._execute_tool_call(tool_name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

                current_iteration += 1
                continue
            
            final_response = assistant_message.content
            messages.append({"role": "assistant", "content": final_response})
            break

        if current_iteration >= iterations_limit and not final_response:
            final_choice = self.client.invoke_chat(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = final_choice.choices[0].message.content
            messages.append({"role": "assistant", "content": final_response})

        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_response))
        return final_response

    def add_tool(self, tool: Tool) -> None:
        """便捷方法：将工具注册到当前Agent"""
        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        return self.tool_registry is not None

    def stream_run(
        self,
        user_input: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        流式运行Agent，支持工具调用

        Args:
            user_input: 用户输入
            max_tool_iterations: 最大工具调用迭代次数
            tool_choice: 工具选择策略
            **kwargs: 其他参数

        Yields:
            Agent响应片段
        """
        messages = self._build_messages(user_input)
        tool_schemas = self._build_tool_schemas()
        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice
        current_iteration = 0
        final_response = ""

        # 如果没有工具，使用简单的流式调用
        if not tool_schemas:
            for chunk in self.client.stream_chat(messages, **kwargs):
                final_response += chunk
                yield chunk

            self.add_message(Message("user", user_input))
            self.add_message(Message("assistant", final_response))
            return

        # 有工具时，使用迭代式调用（类似run方法）
        while current_iteration < iterations_limit:
            response = self.client.invoke_chat(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            assistant_message = response.choices[0].message or {"role": "assistant", "content": ""}
            tool_calls = list(assistant_message.tool_calls or [])

            if tool_calls:
                # 有工具调用，执行工具并继续
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

                    current_iteration += 1
                continue

            # 没有工具调用，流式输出最终响应
            final_response = assistant_message.content or ""
            messages.append({"role": "assistant", "content": final_response})
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= iterations_limit and not final_response:
            final_choice = self.client.invoke_chat(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = final_choice.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": final_response})

        # 保存到历史记录
        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_response))

        # 流式输出最终响应
        for chunk in final_response:
            yield chunk