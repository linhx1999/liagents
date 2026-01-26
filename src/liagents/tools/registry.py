from typing import Optional, Any
from .base import Tool


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        注册Tool对象

        Args:
            tool: Tool实例
        """
        if tool.name in self._tools:
            print(f"警告：工具 '{tool.name}' 已存在，将被覆盖。")

        self._tools[tool.name] = tool
        print(f"工具 '{tool.name}' 已注册。")

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"工具 '{name}' 已注销。")
        else:
            print(f"警告：工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取Tool对象"""
        return self._tools.get(name)

    def execute_tool(self, name: str, parameters: dict[str, Any]) -> str:
        """
        执行工具

        Args:
            name: 工具名称
            parameters: 输入参数

        Returns:
            工具执行结果
        """
        # 优先查找Tool对象
        if name in self._tools:
            tool = self._tools[name]
            try:
                # 直接传入字典参数
                return tool.run(parameters)
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"

        else:
            return f"错误：未找到名为 '{name}' 的工具。"

    def get_tools_description(self) -> str:
        """
        获取所有可用工具的格式化描述字符串

        Returns:
            工具描述字符串，用于构建提示词
        """
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def list_tools(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def get_all_tools(self) -> list[Tool]:
        """获取所有Tool对象"""
        return list(self._tools.values())

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        print("所有工具已清空。")


# 全局工具注册表
tool_registry = ToolRegistry()
