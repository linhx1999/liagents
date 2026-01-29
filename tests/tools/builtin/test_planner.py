"""测试 Planner 工具 (write_todos)"""

import pytest

from liagents.tools.builtin.planner import write_todos


class TestPlannerTool:
    """测试 Planner 工具"""

    def test_tool_name(self):
        """测试工具名称"""
        assert write_todos.name == "write_todos"

    def test_tool_description(self):
        """测试工具描述"""
        assert "规划" in write_todos.description
        assert "追踪" in write_todos.description
        assert "任务" in write_todos.description

    def test_get_parameters(self):
        """测试获取参数"""
        params = write_todos.get_parameters()

        assert len(params) == 1
        assert params[0].name == "todo_list"
        assert params[0].type == "array"
        assert "待办事项" in params[0].description

    def test_run_with_valid_todo_list(self):
        """测试运行有效的待办事项列表"""
        todo_list = [
            {"content": "任务1", "status": "pending"},
            {"content": "任务2", "status": "in_progress"},
            {"content": "任务3", "status": "completed"},
        ]

        result = write_todos.run({"todo_list": todo_list})

        assert str(todo_list) in result

    def test_run_with_empty_list(self):
        """测试运行空列表"""
        result = write_todos.run({"todo_list": []})

        assert "[]" in result

    def test_run_with_single_item(self):
        """测试运行单个待办事项"""
        todo_list = [{"content": "唯一任务", "status": "pending"}]

        result = write_todos.run({"todo_list": todo_list})

        assert "唯一任务" in result
        assert "pending" in result

    def test_run_missing_required_param(self):
        """测试缺少必需参数"""
        result = write_todos.run({})

        # 工具会尝试执行但可能失败
        assert isinstance(result, str)

    def test_to_schema(self):
        """测试转换为 schema"""
        schema = write_todos.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "write_todos"
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"

    def test_to_dict(self):
        """测试转换为字典"""
        result = write_todos.to_dict()

        assert result["name"] == "write_todos"
        assert "parameters" in result
        assert len(result["parameters"]) == 1

    def test_validate_parameters_valid(self):
        """测试有效参数验证"""
        result = write_todos.validate_parameters({"todo_list": []})
        assert result is True

    def test_validate_parameters_missing_required(self):
        """测试缺少必需参数验证"""
        result = write_todos.validate_parameters({})
        assert result is False

    def test_run_with_all_statuses(self):
        """测试所有状态类型"""
        todo_list = [
            {"content": "待办任务", "status": "pending"},
            {"content": "进行中任务", "status": "in_progress"},
            {"content": "已完成任务", "status": "completed"},
        ]

        result = write_todos.run({"todo_list": todo_list})

        assert "pending" in result
        assert "in_progress" in result
        assert "completed" in result

    def test_run_with_custom_status(self):
        """测试自定义状态"""
        todo_list = [{"content": "特殊任务", "status": "custom_status"}]

        result = write_todos.run({"todo_list": todo_list})

        assert "custom_status" in result

    def test_run_with_nested_structure(self):
        """测试嵌套结构"""
        todo_list = [
            {
                "content": "复杂任务",
                "status": "pending",
                "priority": "high",
                "subtasks": [
                    {"content": "子任务1", "status": "pending"},
                ],
            }
        ]

        result = write_todos.run({"todo_list": todo_list})

        assert "复杂任务" in result
        assert "子任务1" in result

    def test_str_representation(self):
        """测试字符串表示"""
        result = str(write_todos)

        assert "write_todos" in result
        assert "Tool" in result


class TestPlannerToolEdgeCases:
    """测试 Planner 工具边界情况"""

    def test_run_with_invalid_input(self):
        """测试无效输入"""
        # 工具期望 list[dict[str, str]]，传入其他类型
        result = write_todos.run({"todo_list": "not a list"})

        # 可能会失败或返回错误结果
        assert isinstance(result, str)

    def test_run_with_malformed_dict(self):
        """测试格式错误的字典"""
        # 传入不包含 content 或 status 的字典
        todo_list = [
            {"content": "正常任务", "status": "pending"},
            {"invalid": "数据"},  # 缺少必需字段
        ]

        result = write_todos.run({"todo_list": todo_list})

        assert isinstance(result, str)
        # 仍然应该包含有效任务
        assert "正常任务" in result

    def test_run_with_very_long_list(self):
        """测试很长的列表"""
        todo_list = [{"content": f"任务{i}", "status": "pending"} for i in range(100)]

        result = write_todos.run({"todo_list": todo_list})

        assert "任务0" in result
        assert "任务99" in result
