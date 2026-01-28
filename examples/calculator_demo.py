from dotenv import load_dotenv

load_dotenv()

from liagents.tools.registry import ToolRegistry
from liagents.tools.builtin.calculator import python_calculator
from liagents.agents.react_agent import ReActAgent
from liagents.agents.openai_func_call_agent import OpenAIFuncCallAgent
from liagents.core.client import Client


def test_calculator_tool():
    """测试自定义计算器工具"""

    # 创建包含计算器的注册表
    registry = ToolRegistry()
    registry.register_tool(python_calculator)

    print("测试自定义计算器工具\n")

    # 简单测试用例
    test_cases = [
        "2 + 3",  # 基本加法
        "10 - 4",  # 基本减法
        "5 * 6",  # 基本乘法
        "15 / 3",  # 基本除法
        "sqrt(16)",  # 平方根
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("python_calculator", {"expression": expression})
        print(f"结果: {result}\n")

    print("======\n")


def test_agent_without_tool(user_question: str):
    client = Client()
    agent = ReActAgent(client=client)

    print("测试不带工具的 ReActAgent:\n")

    print(f"用户问题: {user_question}")

    print("\nReActAgent 的回答:")
    response = agent.run(user_question)
    print(response)
    print("======\n")


def test_agent_with_tool(user_question: str):
    # 创建LLM客户端
    client = Client()
    agent = ReActAgent(client=client)

    agent.add_tool(python_calculator)

    print("与 ReActAgent 集成测试:")

    print(f"用户问题: {user_question}")

    print("\nReActAgent 的回答:")
    response = agent.run(user_question)
    print(response)
    print("======\n")


def test_openai_func_call_agent_with_tool(user_question: str):
    """测试 OpenAIFuncCallAgent（使用原生 Function Calling）"""
    client = Client()
    agent = OpenAIFuncCallAgent(
        name="OpenAIFuncCallAgent",
        client=client,
        system_prompt="你是一个有帮助的助手。",
    )

    agent.add_tool(python_calculator)

    print("与 OpenAIFuncCallAgent 集成测试 (原生 Function Calling):")

    print(f"用户问题: {user_question}")

    print("\nOpenAIFuncCallAgent 的回答:")
    response = agent.run(user_question)
    print(response)
    print("======\n")


if __name__ == "__main__":
    test_calculator_tool()

    user_question = "请帮我计算 sqrt(16) + 2 * 3"
    test_agent_without_tool(user_question)
    test_agent_with_tool(user_question)
    test_openai_func_call_agent_with_tool(user_question)
