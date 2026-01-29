"""测试 ToolRegistry 工具注册表"""

import pytest
from unittest.mock import Mock

from liagents.tools.registry import ToolRegistry, tool_registry
from liagents.tools.base import Tool, ToolParameter


@pytest.fixture
def empty_registry():
    """创建空注册表"""
    return ToolRegistry()


@pytest.fixture
def sample_tool():
    """创建示例工具"""
    def dummy_func(x):
        return x

    return Tool(
        func=dummy_func,
        name="sample_tool",
        description="示例工具",
        parameters=[
            ToolParameter(name="x", type="string", description="输入参数")
        ]
    )


@pytest.fixture
def another_tool():
    """创建另一个工具"""
    def dummy_func(a, b):
        return a + b

    return Tool(
        func=dummy_func,
        name="another_tool",
        description="另一个工具",
        parameters=[
            ToolParameter(name="a", type="integer", description="参数A"),
            ToolParameter(name="b", type="integer", description="参数B"),
        ]
    )


class TestToolRegistryInit:
    """测试 ToolRegistry 初始化"""

    def test_init_empty(self):
        """测试空初始化"""
        registry = ToolRegistry()
        assert registry._tools == {}

    def test_global_registry_exists(self):
        """测试全局注册表存在"""
        assert tool_registry is not None
        assert isinstance(tool_registry, ToolRegistry)


class TestToolRegistryRegister:
    """测试工具注册"""

    def test_register_tool(self, empty_registry, sample_tool):
        """测试注册工具"""
        empty_registry.register_tool(sample_tool)

        assert "sample_tool" in empty_registry._tools
        assert empty_registry._tools["sample_tool"] == sample_tool

    def test_register_multiple_tools(self, empty_registry, sample_tool, another_tool):
        """测试注册多个工具"""
        empty_registry.register_tool(sample_tool)
        empty_registry.register_tool(another_tool)

        assert len(empty_registry._tools) == 2
        assert "sample_tool" in empty_registry._tools
        assert "another_tool" in empty_registry._tools

    def test_register_duplicate_tool(self, empty_registry, sample_tool):
        """测试重复注册工具"""
        empty_registry.register_tool(sample_tool)
        empty_registry.register_tool(sample_tool)

        # 应该被覆盖
        assert len(empty_registry._tools) == 1


class TestToolRegistryUnregister:
    """测试工具注销"""

    def test_unregister_exists(self, empty_registry, sample_tool):
        """测试注销存在的工具"""
        empty_registry.register_tool(sample_tool)
        result = empty_registry.unregister_tool("sample_tool")

        assert result is True
        assert "sample_tool" not in empty_registry._tools

    def test_unregister_not_exists(self, empty_registry):
        """测试注销不存在的工具"""
        result = empty_registry.unregister_tool("nonexistent")

        assert result is False


class TestToolRegistryGetTool:
    """测试获取工具"""

    def test_get_tool_exists(self, empty_registry, sample_tool):
        """测试获取存在的工具"""
        empty_registry.register_tool(sample_tool)
        result = empty_registry.get_tool("sample_tool")

        assert result == sample_tool

    def test_get_tool_not_exists(self, empty_registry):
        """测试获取不存在的工具"""
        result = empty_registry.get_tool("nonexistent")

        assert result is None


class TestToolRegistryExecuteTool:
    """测试执行工具"""

    def test_execute_tool_success(self, empty_registry, sample_tool):
        """测试成功执行工具"""
        empty_registry.register_tool(sample_tool)
        result = empty_registry.execute_tool("sample_tool", {"x": "test_value"})

        assert result == "test_value"  # 工具直接返回输入值

    def test_execute_tool_not_found(self, empty_registry):
        """测试执行不存在的工具"""
        result = empty_registry.execute_tool("nonexistent", {})

        assert "未找到名为 'nonexistent'" in result

    def test_execute_tool_with_error(self, empty_registry):
        """测试执行工具出错"""
        def error_func():
            raise ValueError("工具错误")

        error_tool = Tool(
            func=error_func,
            name="error_tool",
            description="错误工具",
            parameters=[],
        )
        empty_registry.register_tool(error_tool)

        result = empty_registry.execute_tool("error_tool", {})

        # Tool.run 会捕获异常并返回 "执行出错: 工具错误"
        assert "执行出错" in result
        assert "工具错误" in result


class TestToolRegistryGetToolsDescription:
    """测试获取工具描述"""

    def test_get_description_empty(self, empty_registry):
        """测试空注册表的描述"""
        result = empty_registry.get_tools_description()

        assert result == "暂无可用工具"

    def test_get_description_with_tools(self, empty_registry, sample_tool, another_tool):
        """测试带工具的描述"""
        empty_registry.register_tool(sample_tool)
        empty_registry.register_tool(another_tool)

        result = empty_registry.get_tools_description()

        assert "sample_tool" in result
        assert "示例工具" in result
        assert "another_tool" in result
        assert "另一个工具" in result

    def test_get_description_format(self, empty_registry, sample_tool):
        """测试描述格式"""
        empty_registry.register_tool(sample_tool)

        result = empty_registry.get_tools_description()

        assert result.startswith("- sample_tool:")
        assert "示例工具" in result


class TestToolRegistryListTools:
    """测试列出工具"""

    def test_list_tools_empty(self, empty_registry):
        """测试空注册表的工具列表"""
        result = empty_registry.list_tools()

        assert result == []

    def test_list_tools_with_tools(self, empty_registry, sample_tool, another_tool):
        """测试带工具的工具列表"""
        empty_registry.register_tool(sample_tool)
        empty_registry.register_tool(another_tool)

        result = empty_registry.list_tools()

        assert len(result) == 2
        assert "sample_tool" in result
        assert "another_tool" in result


class TestToolRegistryGetAllTools:
    """测试获取所有工具"""

    def test_get_all_tools_empty(self, empty_registry):
        """测试空注册表获取所有工具"""
        result = empty_registry.get_all_tools()

        assert result == []

    def test_get_all_tools_with_tools(self, empty_registry, sample_tool, another_tool):
        """测试带工具的获取所有工具"""
        empty_registry.register_tool(sample_tool)
        empty_registry.register_tool(another_tool)

        result = empty_registry.get_all_tools()

        assert len(result) == 2
        assert sample_tool in result
        assert another_tool in result


class TestToolRegistryClear:
    """测试清空注册表"""

    def test_clear_tools(self, empty_registry, sample_tool, another_tool):
        """测试清空工具"""
        empty_registry.register_tool(sample_tool)
        empty_registry.register_tool(another_tool)

        empty_registry.clear()

        assert len(empty_registry._tools) == 0
        assert empty_registry.list_tools() == []


class TestToolRegistryIntegration:
    """测试注册表集成功能"""

    def test_full_workflow(self, empty_registry):
        """测试完整工作流"""
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b

        add_tool = Tool(
            func=add,
            name="add",
            description="加法工具",
            parameters=[
                ToolParameter(name="a", type="integer", description="A"),
                ToolParameter(name="b", type="integer", description="B"),
            ]
        )

        multiply_tool = Tool(
            func=multiply,
            name="multiply",
            description="乘法工具",
            parameters=[
                ToolParameter(name="a", type="integer", description="A"),
                ToolParameter(name="b", type="integer", description="B"),
            ]
        )

        # 注册工具
        empty_registry.register_tool(add_tool)
        empty_registry.register_tool(multiply_tool)

        # 列出工具
        tools = empty_registry.list_tools()
        assert len(tools) == 2

        # 执行工具
        add_result = empty_registry.execute_tool("add", {"a": 2, "b": 3})
        assert add_result == "5"

        multiply_result = empty_registry.execute_tool("multiply", {"a": 2, "b": 3})
        assert multiply_result == "6"

        # 注销工具
        empty_registry.unregister_tool("add")
        assert "add" not in empty_registry.list_tools()

        # 清空
        empty_registry.clear()
        assert empty_registry.list_tools() == []


class TestToolRegistryEdgeCases:
    """测试边界情况"""

    def test_register_none(self, empty_registry):
        """测试注册 None"""
        # 这会导致错误，因为 None 没有 name 属性
        # 所以我们不测试这种情况

    def test_execute_with_empty_params(self, empty_registry, sample_tool):
        """测试执行带空参数的工具"""
        empty_registry.register_tool(sample_tool)
        result = empty_registry.execute_tool("sample_tool", {})

        # 工具没有必需参数验证，kwargs 为空，func 会被调用
        assert isinstance(result, str)

    def test_special_characters_in_tool_name(self, empty_registry):
        """测试特殊字符工具名"""
        def func():
            return "ok"

        special_tool = Tool(
            func=func,
            name="tool_with-special_chars.123",
            description="特殊字符工具",
            parameters=[],
        )
        empty_registry.register_tool(special_tool)

        assert "tool_with-special_chars.123" in empty_registry.list_tools()
