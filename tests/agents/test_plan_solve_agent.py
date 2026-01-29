"""测试 PlanSolveAgent"""

import pytest
import os
from unittest.mock import Mock
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

# 设置必需的环境变量
os.environ["MODEL"] = "test-model"
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["OPENAI_BASE_URL"] = "https://test.example.com"

from liagents.agents.plan_solve_agent import PlanSolveAgent
from liagents.core.client import Client
from liagents.core.config import Config
from liagents.core.message import Message
from liagents.tools.registry import tool_registry


# ========== 测试 Fixtures ==========


@pytest.fixture(autouse=True)
def cleanup_tool_registry():
    """每个测试后清理工具注册表"""
    yield
    tool_registry.clear()


@pytest.fixture
def mock_client():
    """创建 Mock Client 实例"""
    client = Mock(spec=Client)
    client.chat_with_tools = Mock()
    return client


@pytest.fixture
def mock_config():
    """创建 Mock Config 实例"""
    return Config()


@pytest.fixture
def plan_solve_agent(mock_client):
    """创建 PlanSolveAgent 实例（使用默认配置）"""
    agent = PlanSolveAgent(
        name="test_plan_agent",
        client=mock_client,
        system_prompt="你是一个测试助手",
    )
    yield agent
    # 测试后清理
    agent.tool_registry.clear()


@pytest.fixture
def plan_solve_agent_with_custom_prompt(mock_client):
    """创建带自定义系统提示词的 PlanSolveAgent 实例"""
    return PlanSolveAgent(
        name="custom_plan_agent",
        client=mock_client,
        system_prompt="自定义提示词",
    )


@pytest.fixture
def mock_chat_completion_without_tools():
    """创建不带工具调用的模拟响应"""
    message = ChatCompletionMessage(
        role="assistant",
        content="任务已完成！",
    )
    choice = Choice(index=0, message=message, finish_reason="stop")
    completion = ChatCompletion(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[choice],
        object="chat.completion",
    )
    return completion


@pytest.fixture
def mock_chat_completion_with_todo_call():
    """创建带 write_todos 工具调用的模拟响应"""
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        type="function",
        function=Function(
            name="write_todos",
            arguments='{"todo_list": [{"content": "步骤1", "status": "pending"}, {"content": "步骤2", "status": "pending"}]}',
        ),
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    choice = Choice(index=0, message=message, finish_reason="tool_calls")
    completion = ChatCompletion(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[choice],
        object="chat.completion",
    )
    return completion


# ========== 初始化测试 ==========


class TestPlanSolveAgentInit:
    """测试 PlanSolveAgent 初始化"""

    def test_init_basic(self, mock_client):
        """测试基本初始化"""
        agent = PlanSolveAgent(
            name="test_agent",
            client=mock_client,
            system_prompt="测试提示词",
        )

        assert agent.name == "test_agent"
        assert agent.client == mock_client
        assert agent.system_prompt == "测试提示词"
        assert agent.tool_registry is not None
        assert agent._history == []

    def test_init_without_name(self, mock_client):
        """测试默认名称"""
        agent = PlanSolveAgent(client=mock_client)

        assert agent.name == "PlanSolveAgent"

    def test_init_with_custom_max_iterations(self, mock_client):
        """测试自定义 max_tool_iterations"""
        agent = PlanSolveAgent(
            name="test_agent",
            client=mock_client,
            max_tool_iterations=5,
        )

        assert agent.max_tool_iterations == 5

    def test_init_with_default_tool_choice(self, mock_client):
        """测试默认 tool_choice"""
        agent = PlanSolveAgent(name="test_agent", client=mock_client)

        assert agent.tool_choice == "auto"


# ========== 默认工具测试 ==========


class TestDefaultTools:
    """测试默认工具配置"""

    def test_write_todos_tool_registered(self, plan_solve_agent):
        """测试 write_todos 工具已注册"""
        tools = plan_solve_agent.list_tools()

        assert "write_todos" in tools

    def test_has_write_todos_tool(self, plan_solve_agent):
        """测试有 write_todos 工具"""
        assert plan_solve_agent.has_tools() is True

    def test_get_write_todos_tool(self, plan_solve_agent):
        """测试获取 write_todos 工具"""
        tool = plan_solve_agent.tool_registry.get_tool("write_todos")

        assert tool is not None
        assert tool.name == "write_todos"

    def test_can_remove_write_todos_tool(self, plan_solve_agent):
        """测试可以移除 write_todos 工具"""
        result = plan_solve_agent.remove_tool("write_todos")

        assert result is True
        assert "write_todos" not in plan_solve_agent.list_tools()


# ========== 继承方法测试 ==========


class TestInheritedMethods:
    """测试从 OpenAIFuncCallAgent 继承的方法"""

    def test_add_tool(self, plan_solve_agent):
        """测试添加工具"""
        from liagents.tools.base import Tool

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "extra_tool"
        mock_tool.description = "额外工具"

        plan_solve_agent.add_tool(mock_tool)

        assert "extra_tool" in plan_solve_agent.list_tools()

    def test_list_tools_includes_write_todos(self, plan_solve_agent):
        """测试列出工具包含 write_todos"""
        tools = plan_solve_agent.list_tools()

        assert len(tools) >= 1
        assert "write_todos" in tools

    def test_run_inherits_from_parent(
        self, plan_solve_agent, mock_chat_completion_without_tools
    ):
        """测试 run 方法继承正常"""
        plan_solve_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        response = plan_solve_agent.run("测试任务")

        assert response == "任务已完成！"

    def test_run_with_history(self, plan_solve_agent):
        """测试带历史记录的运行"""
        plan_solve_agent._history.append(Message("user", "之前的消息"))
        plan_solve_agent._history.append(Message("assistant", "之前的回复"))

        mock_message = ChatCompletionMessage(role="assistant", content="新的回复")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[mock_choice],
            object="chat.completion",
        )
        plan_solve_agent.client.chat_with_tools.return_value = mock_completion

        response = plan_solve_agent.run("新消息")

        # 验证响应正确
        assert response == "新的回复"

        # 验证历史记录更新：原有2条 + user消息 + assistant响应 = 4条
        assert len(plan_solve_agent._history) == 4

    def test_stream_run_inherits_from_parent(self, plan_solve_agent):
        """测试 stream_run 方法继承正常"""
        plan_solve_agent.run = Mock(return_value="流式回复")

        chunks = list(plan_solve_agent.stream_run("测试输入"))

        assert len(chunks) == 1
        assert chunks[0] == "流式回复"


# ========== 运行测试 ==========


class TestRun:
    """测试 run 方法"""

    def test_run_without_tools(
        self, plan_solve_agent, mock_chat_completion_without_tools
    ):
        """测试不带工具的运行"""
        plan_solve_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        response = plan_solve_agent.run("你好")

        assert response == "任务已完成！"
        assert len(plan_solve_agent._history) == 2

    def test_run_with_tool_call(
        self, plan_solve_agent, mock_chat_completion_with_todo_call
    ):
        """测试带工具调用的运行"""
        # 最终响应
        final_message = ChatCompletionMessage(role="assistant", content="任务完成，这是最终答案")
        final_choice = Choice(index=0, message=final_message, finish_reason="stop")
        final_completion = ChatCompletion(
            id="test-id-final",
            created=1234567891,
            model="test-model",
            choices=[final_choice],
            object="chat.completion",
        )

        plan_solve_agent.client.chat_with_tools = Mock(
            side_effect=[
                mock_chat_completion_with_todo_call,  # 第一次:工具调用
                final_completion,  # 第二次:最终响应
            ]
        )

        response = plan_solve_agent.run("执行任务")

        assert response == "任务完成，这是最终答案"
        assert plan_solve_agent.client.chat_with_tools.call_count == 2

    def test_run_with_custom_max_iterations(self, plan_solve_agent):
        """测试自定义 max_tool_iterations"""
        mock_message = ChatCompletionMessage(role="assistant", content="任务已完成！")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[mock_choice],
            object="chat.completion",
        )
        plan_solve_agent.client.chat_with_tools.return_value = mock_completion

        response = plan_solve_agent.run("测试", max_tool_iterations=5)

        assert response == "任务已完成！"

    def test_run_passes_kwargs(
        self, plan_solve_agent, mock_chat_completion_without_tools
    ):
        """测试传递额外参数"""
        plan_solve_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        plan_solve_agent.run("测试", temperature=0.5, max_tokens=100)

        call_kwargs = plan_solve_agent.client.chat_with_tools.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5


# ========== 边界情况测试 ==========


class TestEdgeCases:
    """测试边界情况"""

    def test_run_with_empty_input(self, plan_solve_agent):
        """测试空输入"""
        mock_message = ChatCompletionMessage(role="assistant", content="")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[mock_choice],
            object="chat.completion",
        )
        plan_solve_agent.client.chat_with_tools = Mock(return_value=mock_completion)

        response = plan_solve_agent.run("")

        assert response == ""

    def test_add_multiple_tools(self, plan_solve_agent):
        """测试添加多个工具"""
        from liagents.tools.base import Tool

        mock_tool1 = Mock(spec=Tool)
        mock_tool1.name = "tool1"
        mock_tool1.description = "工具1"

        mock_tool2 = Mock(spec=Tool)
        mock_tool2.name = "tool2"
        mock_tool2.description = "工具2"

        plan_solve_agent.add_tool(mock_tool1)
        plan_solve_agent.add_tool(mock_tool2)

        tools = plan_solve_agent.list_tools()
        assert "tool1" in tools
        assert "tool2" in tools
        assert "write_todos" in tools  # 默认工具应仍存在

    def test_remove_all_tools(self, plan_solve_agent):
        """测试移除所有工具"""
        plan_solve_agent.remove_tool("write_todos")
        plan_solve_agent.remove_tool("think")

        # has_tools() 检查 tool_registry 是否为 None，不是检查是否有工具
        # 所以这里检查 list_tools() 是否为空即可
        assert plan_solve_agent.list_tools() == []
        assert len(plan_solve_agent.tool_registry.list_tools()) == 0
