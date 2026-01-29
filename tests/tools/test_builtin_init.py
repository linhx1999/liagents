"""测试 builtin 工具包初始化"""

import pytest


class TestBuiltinInit:
    """测试 builtin 包初始化"""

    def test_python_calculator_imported(self):
        """测试计算器工具已导入"""
        from liagents.tools.builtin import python_calculator

        assert python_calculator is not None
        assert python_calculator.name == "python_calculator"

    def test_think_imported(self):
        """测试 think 工具已导入"""
        from liagents.tools.builtin import think

        assert think is not None
        assert think.name == "think"

    def test_write_todos_imported(self):
        """测试 write_todos 工具已导入"""
        from liagents.tools.builtin import write_todos

        assert write_todos is not None
        assert write_todos.name == "write_todos"

    def test_tavily_search_imported(self):
        """测试 tavily_search 工具已导入"""
        from liagents.tools.builtin import tavily_search

        assert tavily_search is not None
        assert tavily_search.name == "tavily_search"

    def test_all_in_init(self):
        """测试 __all__ 导出"""
        from liagents.tools.builtin import __all__

        from liagents.tools.builtin import (
            python_calculator,
            think,
            write_todos,
            tavily_search,
        )

        assert python_calculator in __all__
        assert think in __all__
        assert write_todos in __all__
        assert tavily_search in __all__


class TestBuiltinToolsAvailability:
    """测试 builtin 工具可用性"""

    def test_calculator_is_tool(self):
        """测试计算器是 Tool 实例"""
        from liagents.tools.builtin import python_calculator
        from liagents.tools.base import Tool

        assert isinstance(python_calculator, Tool)

    def test_think_is_tool(self):
        """测试 think 是 Tool 实例"""
        from liagents.tools.builtin import think
        from liagents.tools.base import Tool

        assert isinstance(think, Tool)

    def test_write_todos_is_tool(self):
        """测试 write_todos 是 Tool 实例"""
        from liagents.tools.builtin import write_todos
        from liagents.tools.base import Tool

        assert isinstance(write_todos, Tool)

    def test_tavily_search_is_tool(self):
        """测试 tavily_search 是 Tool 实例"""
        from liagents.tools.builtin import tavily_search
        from liagents.tools.base import Tool

        assert isinstance(tavily_search, Tool)


class TestBuiltinToolsExecution:
    """测试 builtin 工具执行"""

    def test_calculator_execution(self):
        """测试计算器执行"""
        from liagents.tools.builtin import python_calculator

        result = python_calculator.run({"expression": "2 + 2"})
        assert result == "4"

    def test_think_execution(self):
        """测试 think 执行"""
        from liagents.tools.builtin import think

        result = think.run({"thinking": "测试思考"})
        assert result == "测试思考"

    def test_write_todos_execution(self):
        """测试 write_todos 执行"""
        from liagents.tools.builtin import write_todos

        todo_list = [{"content": "测试任务", "status": "pending"}]
        result = write_todos.run({"todo_list": todo_list})
        assert "测试任务" in result
