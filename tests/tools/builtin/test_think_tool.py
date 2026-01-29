"""测试 Think 工具"""

import pytest

from liagents.tools.builtin.think_tool import think


class TestThinkTool:
    """测试 Think 工具"""

    def test_tool_name(self):
        """测试工具名称"""
        assert think.name == "think"

    def test_tool_description(self):
        """测试工具描述"""
        assert "推理" in think.description
        assert "分析" in think.description

    def test_get_parameters(self):
        """测试获取参数"""
        params = think.get_parameters()

        assert len(params) == 1
        assert params[0].name == "thinking"
        assert params[0].type == "string"
        assert "推理内容" in params[0].description

    def test_run_with_simple_text(self):
        """测试运行简单文本"""
        thinking = "这是一个简单的思考过程"
        result = think.run({"thinking": thinking})

        assert result == thinking

    def test_run_with_empty_string(self):
        """测试运行空字符串"""
        result = think.run({"thinking": ""})

        assert result == ""

    def test_run_with_long_text(self):
        """测试运行长文本"""
        long_thinking = "这是一个很长的思考过程" * 100
        result = think.run({"thinking": long_thinking})

        assert result == long_thinking

    def test_run_with_special_characters(self):
        """测试运行带特殊字符的文本"""
        special_text = "思考内容\n\t!@#$%^&*()"
        result = think.run({"thinking": special_text})

        assert result == special_text

    def test_run_with_unicode(self):
        """测试运行 Unicode 文本"""
        unicode_text = "中文思考 🎉 αβγ 🚀"
        result = think.run({"thinking": unicode_text})

        assert result == unicode_text

    def test_run_with_newlines(self):
        """测试运行带换行的文本"""
        multiline = "第一行\n第二行\n第三行"
        result = think.run({"thinking": multiline})

        assert result == multiline

    def test_run_missing_required_param(self):
        """测试缺少必需参数"""
        result = think.run({})

        # 工具会尝试执行但 kwargs 为空
        assert isinstance(result, str)

    def test_to_schema(self):
        """测试转换为 schema"""
        schema = think.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "think"
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"

    def test_to_dict(self):
        """测试转换为字典"""
        result = think.to_dict()

        assert result["name"] == "think"
        assert "parameters" in result
        assert len(result["parameters"]) == 1

    def test_validate_parameters_valid(self):
        """测试有效参数验证"""
        result = think.validate_parameters({"thinking": "some thought"})
        assert result is True

    def test_validate_parameters_missing_required(self):
        """测试缺少必需参数验证"""
        result = think.validate_parameters({})
        assert result is False

    def test_str_representation(self):
        """测试字符串表示"""
        result = str(think)

        assert "think" in result
        assert "Tool" in result


class TestThinkToolUseCases:
    """测试 Think 工具使用场景"""

    def test_problem_analysis(self):
        """测试问题分析场景"""
        analysis = """问题分析：
1. 核心需求是...
2. 可能的解决方案有...
3. 最佳方案是...
"""
        result = think.run({"thinking": analysis})

        assert "问题分析" in result

    def test_solution_evaluation(self):
        """测试方案评估场景"""
        evaluation = """方案评估：
- 优点：实现简单
- 缺点：性能可能不佳
- 结论：可以接受
"""
        result = think.run({"thinking": evaluation})

        assert "优点" in result

    def test_step_planning(self):
        """测试步骤规划场景"""
        planning = """步骤规划：
1. 首先完成X
2. 然后处理Y
3. 最后验证Z
"""
        result = think.run({"thinking": planning})

        assert "步骤规划" in result

    def test_self_reflection(self):
        """测试自我反思场景"""
        reflection = """自我反思：
- 之前的方法存在什么问题
- 如何改进
- 下次需要注意什么
"""
        result = think.run({"thinking": reflection})

        assert "自我反思" in result


class TestThinkToolEdgeCases:
    """测试 Think 工具边界情况"""

    def test_run_with_whitespace_only(self):
        """测试只包含空白的输入"""
        result = think.run({"thinking": "   \t\n  "})

        assert result == "   \t\n  "

    def test_run_with_json_like_content(self):
        """测试 JSON -like 内容"""
        json_content = '{"key": "value", "nested": {"inner": "data"}}'
        result = think.run({"thinking": json_content})

        assert result == json_content

    def test_run_with_code_snippet(self):
        """测试代码片段"""
        code = """def example():
    return 'hello'
"""
        result = think.run({"thinking": code})

        assert "def example" in result
