from typing import Any, Optional, Iterator, TYPE_CHECKING
import re
import json

from ..core.agent import Agent
from ..core.client import Client
from ..core.config import Config
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """简单的对话Agent，支持可选的工具调用"""

    def __init__(
        self,
        name: str,
        client: Client,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True,
    ):
        """
        初始化SimpleAgent

        Args:
            name: Agent名称
            client: LLM客户端实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            enable_tool_calling: 是否启用工具调用（只有在提供tool_registry时生效）
        """
        super().__init__(name, client, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = (self.system_prompt or "你是一个有用的AI助手。").strip()

        if not self.enable_tool_calling or not self.tool_registry:
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
            result = tool.run(**arguments)
            return f"工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"工具调用失败：{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        import json

        param_dict = {}

        # 尝试解析JSON格式
        if parameters.strip().startswith("{"):
            try:
                param_dict = json.loads(parameters)
                # JSON解析成功，进行类型转换
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                # JSON解析失败，继续使用其他方式
                pass

        if "=" in parameters:
            # 格式: key=value 或 action=search,query=Python
            if "," in parameters:
                # 多个参数：action=search,query=Python,limit=3
                pairs = parameters.split(",")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数：key=value
                key, value = parameters.split("=", 1)
                param_dict[key.strip()] = value.strip()

            # 类型转换
            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # 智能推断action（如果没有指定）
            if "action" not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        else:
            # 直接传入参数，根据工具类型智能推断
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict

    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """
        根据工具的参数定义转换参数类型

        Args:
            tool_name: 工具名称
            param_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        # 获取工具的参数定义
        try:
            tool_params = tool.get_parameters()
        except:
            return param_dict

        # 创建参数类型映射
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        # 转换参数类型
        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    if param_type == "number" or param_type == "integer":
                        # 转换为数字
                        if isinstance(value, str):
                            converted_dict[key] = (
                                float(value) if param_type == "number" else int(value)
                            )
                        else:
                            converted_dict[key] = value
                    elif param_type == "boolean":
                        # 转换为布尔值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ("true", "1", "yes")
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """根据工具类型和参数推断action"""
        if tool_name == "memory":
            if "recall" in param_dict:
                param_dict["action"] = "search"
                param_dict["query"] = param_dict.pop("recall")
            elif "store" in param_dict:
                param_dict["action"] = "add"
                param_dict["content"] = param_dict.pop("store")
            elif "query" in param_dict:
                param_dict["action"] = "search"
            elif "content" in param_dict:
                param_dict["action"] = "add"
        elif tool_name == "rag":
            if "search" in param_dict:
                param_dict["action"] = "search"
                param_dict["query"] = param_dict.pop("search")
            elif "query" in param_dict:
                param_dict["action"] = "search"
            elif "text" in param_dict:
                param_dict["action"] = "add_text"

        return param_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """为简单参数推断完整的参数字典"""
        if tool_name == "rag":
            return {"action": "search", "query": parameters}
        elif tool_name == "memory":
            return {"action": "search", "query": parameters}
        else:
            return {"input": parameters}

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        运行SimpleAgent，支持可选的工具调用

        Args:
            input_text: 用户输入
            max_tool_iterations: 最大工具调用迭代次数（仅在启用工具时有效）
            **kwargs: 其他参数

        Returns:
            Agent响应
        """
        # 构建消息列表
        messages = []

        # 添加系统消息（可能包含工具信息）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用原有逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        # 迭代处理，支持多轮工具调用
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 调用LLM
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(
                        call["tool_name"], call["parameters"]
                    )
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call["original"], "")

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})

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
            final_response = self.llm.invoke(messages, **kwargs)

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到Agent（便利方法）

        Args:
            tool: Tool对象
            auto_expand: 是否自动展开可展开的工具（默认True）

        如果工具是可展开的（expandable=True），会自动展开为多个独立工具
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 直接使用 ToolRegistry 的 register_tool 方法
        # ToolRegistry 会自动处理工具展开
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent

        Args:
            input_text: 用户输入
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

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
