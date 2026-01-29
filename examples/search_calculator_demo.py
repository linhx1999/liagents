from dotenv import load_dotenv

load_dotenv()

from liagents.tools.registry import ToolRegistry
from liagents.tools.builtin.calculator import python_calculator
from liagents.tools.builtin.search_tool_tavily import tavily_search
from liagents.agents.openai_func_call_agent import OpenAIFuncCallAgent
from liagents.core.client import Client


def test_search_tool():
    """测试 Tavily 搜索工具"""

    print("测试 Tavily 搜索工具\n")

    registry = ToolRegistry()
    registry.register_tool(tavily_search)

    test_queries = [
        "Python 有什么特点？",
        "有哪些智能体框架？",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        result = registry.execute_tool("tavily_search", {"query": query})
        print(f"结果:\n{result}\n")

    print("======\n")


def test_openai_func_call_agent_with_tools():
    """测试 OpenAIFuncCallAgent 使用多个工具"""
    print("OpenAIFuncCallAgent 多工具集成测试:")
    client = Client()
    agent = OpenAIFuncCallAgent(
        name="SearchCalcAgent",
        client=client,
        system_prompt="你是一个智能助手，善于使用搜索和计算工具来回答问题。",
    )

    agent.add_tool(tavily_search)
    agent.add_tool(python_calculator)
    user_question = (
        "搜索从上海到伦敦下周二最便宜的直飞航班票价，如果我要为一家四口订票，加上每人 23kg 的额外行李费（查一下该航司的标准），总共需要准备多少预算？"
    )

    print(f"用户问题: {user_question}")
    print("\n回答:")
    response = agent.run(user_question)
    print(response)
    print("======\n")


if __name__ == "__main__":
    test_search_tool()

    test_openai_func_call_agent_with_tools()
