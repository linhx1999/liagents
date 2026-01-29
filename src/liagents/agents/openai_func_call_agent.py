import json
from typing import Iterator, Optional, Union, Any

from ..core.agent import Agent
from ..core.config import Config
from ..core.client import Client
from ..core.message import Message
from ..tools.registry import ToolRegistry


class OpenAIFuncCallAgent(Agent):
    """基于OpenAI原生函数调用机制的Agent"""

    def __init__(
        self,
        name: str = "OpenAIFuncCallAgent",
        client: Client = Client(),
        system_prompt: str = "",
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = ToolRegistry(),
        tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 10,
        debug: bool = False,
    ):
        super().__init__(
            name, client, system_prompt.strip(), config, tool_registry, debug
        )
        self.tool_choice = tool_choice
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
            schema = tool.to_schema()
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

    def _convert_parameter_types(
        self, tool_name: str, param_dict: dict[str, Any]
    ) -> dict[str, Any]:
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
            raise ValueError("未配置工具注册表")

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"未找到工具 '{tool_name}'")

        try:
            typed_arguments = self._convert_parameter_types(tool_name, arguments)
            return tool.run(typed_arguments)
        except Exception as exc:
            return f"工具调用失败：{exc}"

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
        self._debug_print("开始执行", user_input)
        self._debug_print(
            "配置",
            f"iterations_limit: {max_tool_iterations or self.max_tool_iterations}, tool_choice: {tool_choice or self.tool_choice}",
        )

        messages = self._build_messages(user_input)
        tool_schemas = self._build_tool_schemas()
        self._debug_print("工具", f"共 {len(tool_schemas)} 个工具")

        iterations_limit = (
            max_tool_iterations
            if max_tool_iterations is not None
            else self.max_tool_iterations
        )
        effective_tool_choice: Union[str, dict] = (
            tool_choice if tool_choice is not None else self.tool_choice
        )
        current_iteration = 0
        final_response = ""

        while current_iteration < iterations_limit:
            self._debug_print("迭代", f"第 {current_iteration + 1}/{iterations_limit} 次")
            self._debug_print("消息列表", f"共 {len(messages)} 条消息")

            response = self.client.chat_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            assistant_message = response.choices[0].message or {
                "role": "assistant",
                "content": "",
            }
            tool_calls = list(assistant_message.tool_calls or [])

            # 将 assistant 消息添加到历史
            if tool_calls:
                # 有工具调用时，包含 tool_calls 字段
                self._debug_print("工具调用", f"发现 {len(tool_calls)} 个工具调用")
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            tool_call.model_dump() for tool_call in tool_calls
                        ],
                    }
                )
            else:
                # 没有工具调用时，只添加 content
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                    }
                )

            for tool_call in tool_calls:
                # 只处理 function 类型的工具调用
                if tool_call.type != "function":
                    raise ValueError(f"不支持的工具调用类型: {tool_call.type}")

                tool_name = tool_call.function.name
                arguments = self._parse_function_call_arguments(
                    tool_call.function.arguments
                )
                self._debug_print("执行工具", f"{tool_name}, {arguments}")
                result = self._execute_tool_call(tool_name, arguments)
                self._debug_print("工具结果", result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            # 如果有工具调用，继续下一次迭代进行 API 调用
            if tool_calls:
                current_iteration += 1
                continue

            # 没有工具调用时，使用 assistant 的 content 作为最终响应
            final_response = assistant_message.content or ""
            self._debug_print("最终响应", f"\n{final_response}")
            break

        if current_iteration >= iterations_limit and not final_response:
            self._debug_print("迭代超限", "获取最后一次回答")
            final_choice = self.client.chat_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = final_choice.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": final_response})

        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_response))
        return final_response

    def stream_run(
        self,
        user_input: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> Iterator[str]:
        self._debug_print("开始执行(stream)", f"用户输入: {user_input}")
        result = self.run(
            user_input,
            max_tool_iterations=max_tool_iterations,
            tool_choice=tool_choice,
            **kwargs,
        )
        yield result
