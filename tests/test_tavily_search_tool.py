"""测试 TavilySearchTool"""

import pytest
from unittest.mock import MagicMock
from liagents.tools.builtin.search_tool_tavily import tavily_search


# ========== Fixtures ==========


@pytest.fixture
def search_tool():
    """获取搜索工具实例"""
    return tavily_search


@pytest.fixture
def reset_tavily_client():
    """重置 Tavily 客户端缓存"""
    import liagents.tools.builtin.search_tool_tavily as module

    module._tavily_client = None
    yield
    module._tavily_client = None


@pytest.fixture
def mock_tavily_response_with_answer():
    """模拟 Tavily 返回结果（含 AI 答案）"""
    return {
        "answer": "Python 是一种高级编程语言。",
        "results": [
            {
                "title": "Python 官方文档",
                "content": "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。",
                "url": "https://docs.python.org/",
            },
            {
                "title": "Python 教程",
                "content": "本教程介绍了 Python 的基础知识和高级特性。",
                "url": "https://www.python.org/tutorial/",
            },
        ],
    }


@pytest.fixture
def mock_tavily_response_without_answer():
    """模拟 Tavily 返回结果（不含 AI 答案）"""
    return {
        "results": [
            {
                "title": "搜索结果 1",
                "content": "这是搜索结果的详细内容。",
                "url": "https://example.com/1",
            },
            {
                "title": "搜索结果 2",
                "content": "这是另一个搜索结果的详细内容。",
                "url": "https://example.com/2",
            },
        ],
    }


@pytest.fixture
def mock_tavily_empty_response():
    """模拟 Tavily 空结果"""
    return {"results": []}


# ========== 初始化测试 ==========


class TestTavilySearchToolInit:
    """测试 TavilySearchTool 初始化"""

    def test_init_basic(self, search_tool):
        """测试基本初始化"""
        assert search_tool.name == "tavily_search"
        assert "Tavily" in search_tool.description
        assert "搜索" in search_tool.description

    def test_get_parameters(self, search_tool):
        """测试获取参数定义"""
        params = search_tool.get_parameters()

        assert len(params) == 3

        param_names = {p.name for p in params}
        assert param_names == {"query", "max_results", "include_answer"}

        query_param = next(p for p in params if p.name == "query")
        assert query_param.type == "string"
        assert query_param.required is True
        assert "搜索" in query_param.description

        max_results_param = next(p for p in params if p.name == "max_results")
        assert max_results_param.type == "integer"
        assert max_results_param.required is False
        assert max_results_param.default == 5

        include_answer_param = next(p for p in params if p.name == "include_answer")
        assert include_answer_param.type == "boolean"
        assert include_answer_param.required is False
        assert include_answer_param.default is True

    def test_to_schema(self, search_tool):
        """测试转换为 OpenAI function calling schema"""
        result = search_tool.to_schema()

        assert result["type"] == "function"
        assert "function" in result
        assert result["function"]["name"] == "tavily_search"
        assert "parameters" in result["function"]
        assert result["function"]["parameters"]["type"] == "object"
        assert "properties" in result["function"]["parameters"]

        properties = result["function"]["parameters"]["properties"]
        assert "query" in properties
        assert "max_results" in properties
        assert "include_answer" in properties


# ========== 核心功能测试 ==========


class TestTavilySearchFunctionality:
    """测试搜索功能"""

    def test_search_success_with_answer(
        self, search_tool, mock_tavily_response_with_answer, reset_tavily_client
    ):
        """测试成功搜索（包含 AI 答案）"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response_with_answer
        module._tavily_client = mock_client

        result = search_tool.run({"query": "Python 是什么"})

        assert "AI答案" in result or "Python" in result
        assert "Python 官方文档" in result or "相关结果" in result
        mock_client.search.assert_called_once()

    def test_search_success_without_answer(
        self, search_tool, mock_tavily_response_without_answer, reset_tavily_client
    ):
        """测试成功搜索（不包含 AI 答案）"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response_without_answer
        module._tavily_client = mock_client

        result = search_tool.run({"query": "测试查询", "include_answer": False})

        assert "搜索结果 1" in result
        assert "搜索结果 2" in result
        mock_client.search.assert_called_once()

    def test_search_with_custom_max_results(
        self, search_tool, mock_tavily_response_without_answer, reset_tavily_client
    ):
        """测试自定义结果数量"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response_without_answer
        module._tavily_client = mock_client

        result = search_tool.run({"query": "测试", "max_results": 10})

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("max_results") == 10

    def test_search_with_empty_results(
        self, search_tool, mock_tavily_empty_response, reset_tavily_client
    ):
        """测试空结果返回"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_empty_response
        module._tavily_client = mock_client

        result = search_tool.run({"query": "不存在的查询"})

        assert "未找到相关结果" in result or result.strip() != ""


# ========== 错误处理测试 ==========


class TestErrorHandling:
    """测试错误处理"""

    def test_empty_query(self, search_tool):
        """测试空查询"""
        result = search_tool.run({"query": ""})
        assert "错误" in result
        assert "不能为空" in result

    def test_missing_query_param(self, search_tool):
        """测试缺少查询参数"""
        result = search_tool.run({})
        assert "错误" in result or "missing" in result.lower()

    def test_missing_api_key(self, search_tool, mocker):
        """测试未设置 API Key"""
        import importlib
        import liagents.tools.builtin.search_tool_tavily as module

        mocker.patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=True)
        module._tavily_client = None
        importlib.reload(module)

        result = module.tavily_search.run({"query": "测试"})
        assert "TAVILY_API_KEY" in result or "API" in result or "错误" in result

    def test_api_search_failure(self, search_tool, reset_tavily_client):
        """测试 API 调用异常"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("网络错误")
        module._tavily_client = mock_client

        result = search_tool.run({"query": "测试"})

        assert "搜索失败" in result or "错误" in result


# ========== 参数验证测试 ==========


class TestParameterValidation:
    """测试参数验证"""

    def test_validate_parameters_valid(self, search_tool):
        """测试有效参数验证"""
        assert search_tool.validate_parameters({"query": "测试"}) is True

    def test_validate_parameters_with_optional_params(self, search_tool):
        """测试带可选参数的验证"""
        assert (
            search_tool.validate_parameters(
                {"query": "测试", "max_results": 10, "include_answer": False}
            )
            is True
        )

    def test_validate_parameters_missing_required(self, search_tool):
        """测试缺少必需参数"""
        assert search_tool.validate_parameters({}) is False
        assert search_tool.validate_parameters({"max_results": 5}) is False


# ========== 边界情况测试 ==========


class TestEdgeCases:
    """测试边界情况"""

    def test_very_long_query(
        self, search_tool, mock_tavily_empty_response, reset_tavily_client
    ):
        """测试长查询"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_empty_response
        module._tavily_client = mock_client

        long_query = "a" * 1000
        result = search_tool.run({"query": long_query})

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("query") == long_query

    def test_query_with_special_characters(
        self, search_tool, mock_tavily_response_without_answer, reset_tavily_client
    ):
        """测试含特殊字符的查询"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response_without_answer
        module._tavily_client = mock_client

        special_query = "Python! @#$% 中文查询?"
        result = search_tool.run({"query": special_query})

        mock_client.search.assert_called_once()

    def test_zero_max_results(
        self, search_tool, mock_tavily_empty_response, reset_tavily_client
    ):
        """测试 max_results=0"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_empty_response
        module._tavily_client = mock_client

        result = search_tool.run({"query": "测试", "max_results": 0})

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("max_results") == 0

    def test_large_max_results(
        self, search_tool, mock_tavily_response_without_answer, reset_tavily_client
    ):
        """测试较大的 max_results"""
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response_without_answer
        module._tavily_client = mock_client

        result = search_tool.run({"query": "测试", "max_results": 100})

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("max_results") == 100


# ========== 转换方法测试 ==========


class TestConversionMethods:
    """测试转换方法"""

    def test_to_dict(self, search_tool):
        """测试转换为字典"""
        result = search_tool.to_dict()

        assert result["name"] == "tavily_search"
        assert "description" in result
        assert "parameters" in result
        assert isinstance(result["parameters"], list)
        assert len(result["parameters"]) == 3

    def test_str_representation(self, search_tool):
        """测试字符串表示"""
        result = str(search_tool)
        assert "tavily_search" in result
        assert "Tool" in result


# ========== 集成测试示例 ==========


class TestIntegrationExample:
    """集成测试示例"""

    def test_tool_in_registry(self, search_tool):
        """测试工具在注册表中的使用"""
        from liagents.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_tool(search_tool)

        assert "tavily_search" in registry.list_tools()

    def test_execute_via_registry(
        self, search_tool, mock_tavily_response_with_answer, reset_tavily_client
    ):
        """测试通过注册表执行工具"""
        from liagents.tools.registry import ToolRegistry
        import liagents.tools.builtin.search_tool_tavily as module

        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response_with_answer
        module._tavily_client = mock_client

        registry = ToolRegistry()
        registry.register_tool(search_tool)

        result = registry.execute_tool("tavily_search", {"query": "Python"})
        assert isinstance(result, str)
        mock_client.search.assert_called_once()
