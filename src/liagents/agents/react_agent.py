from typing import Optional, Iterator, TYPE_CHECKING
import re
import json

from ..core.agent import Agent
from ..core.client import Client
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class ReActAgent(Agent):
    """ReAct Agent，支持可选的工具调用"""

    def __init__(
        self,
        name: str = "ReActAgent",
        client: Client = Client(),
        system_prompt: str = "",
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = ToolRegistry(),
        debug: bool = False,
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            client: LLM客户端实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            debug: 是否启用调试模式，打印中间状态
        """
        super().__init__(name, client, system_prompt, config, tool_registry, debug)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt.strip()

        if not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = (
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_description}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

        return base_prompt + tools_section

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        解析文本中的工具调用

        支持格式：
        <tool_call>
        {"name": "function_name", "arguments": {"key": "value"}}
        </tool_call>
        """

        tool_calls = []

        # 使用正则表达式匹配 tool_call 标签内容
        pattern = r"<tool_call>\s*\n?({.+?})\s*\n?</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for json_str in matches:
            # 解析 JSON
            call_data = json.loads(json_str)

            tool_name = call_data.get("name", "")
            arguments = call_data.get("arguments", {})

            tool_calls.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "original": json_str,
                }
            )

        return tool_calls

    def _execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"错误：未配置工具注册表"

        try:
            # 获取Tool对象
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"错误：未找到工具 '{tool_name}'"

            # 调用工具
            result = tool.run(arguments)
            return f"工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"工具调用失败：{str(e)}"

    def run(self, user_input: str, max_tool_iterations: int = 5, **kwargs) -> str:
        """
        运行ReActAgent，支持可选的工具调用

        Args:
            user_input: 用户输入
            max_tool_iterations: 最大工具调用迭代次数（仅在启用工具时有效）
            **kwargs: 其他参数

        Returns:
            Agent响应
        """
        self._debug_print("开始执行", f"用户输入: {user_input}")
        self._debug_print("配置", f"max_tool_iterations: {max_tool_iterations}")

        # 构建消息列表
        messages = []

        # 添加系统消息（可能包含工具信息）
        messages.append(
            {"role": "system", "content": self._get_enhanced_system_prompt()}
        )

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_input})

        # 迭代处理，支持多轮工具调用
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            self._debug_print(
                "迭代", f"第 {current_iteration + 1}/{max_tool_iterations} 次"
            )
            self._debug_print("消息列表", f"共 {len(messages)} 条消息")

            # 调用LLM
            response = self.client.chat(messages, **kwargs)
            self._debug_print(
                "LLM响应", response[:500] + "..." if len(response) > 500 else response
            )

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                self._debug_print("工具调用", f"发现 {len(tool_calls)} 个工具调用")
                # 执行所有工具调用并收集结果
                tool_results = []

                for call in tool_calls:
                    self._debug_print(
                        "执行工具", f"工具名: {call['tool_name']}, 参数: {call['arguments']}"
                    )
                    result = self._execute_tool_call(
                        call["tool_name"], call["arguments"]
                    )
                    self._debug_print(
                        "工具结果", result[:300] + "..." if len(result) > 300 else result
                    )
                    tool_results.append(result)
                    # 从响应中移除工具调用标记

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": response})

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append(
                    {
                        "role": "user",
                        "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。",
                    }
                )

                current_iteration += 1
                continue

            # 没有工具调用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            self._debug_print("迭代超限", "获取最后一次回答")
            final_response = self.client.chat(messages, **kwargs)

        self._debug_print(
            "最终响应",
            final_response[:500] + "..."
            if len(final_response) > 500
            else final_response,
        )

        # 保存到历史记录
        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_response))

        return final_response

    def stream_run(self, user_input: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent

        Args:
            user_input: 用户输入
            **kwargs: 其他参数

        Yields:
            Agent响应片段
        """
        # 构建消息列表
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_input})

        # 流式调用LLM
        full_response = ""
        for chunk in self.client.stream_chat(messages, **kwargs):
            full_response += chunk
            yield chunk

        # 保存完整对话到历史记录
        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", full_response))
