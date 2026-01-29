from abc import ABC, abstractmethod
from typing import Any, Optional
from pprint import pformat

from .message import Message
from .client import Client
from .config import Config
from ..tools.registry import ToolRegistry


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        client: Client = Client(),
        system_prompt: str = "",
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = ToolRegistry(),
        debug: bool = False,
    ):
        self.name = name
        self.client = client
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
        self.tool_registry = tool_registry
        self.debug = debug

    def _debug_print(self, title: str, content: Any):
        """打印调试信息"""
        if self.debug:
            print(f"\n=== [{self.name}] {title} ===")
            content_str = pformat(content) if not isinstance(content, str) else content
            print(content_str)

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def add_tool(self, tool) -> None:
        """
        添加工具到Agent

        Args:
            tool: Tool对象
        """
        if self.tool_registry:
            self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具

        Args:
            tool_name: 工具名称

        Returns:
            是否成功移除
        """
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """
        列出所有可用工具

        Returns:
            工具名称列表
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """
        检查是否配置了工具

        Returns:
            是否有工具
        """
        return self.tool_registry is not None

    def __str__(self) -> str:
        return f"Agent(name={self.name})"
