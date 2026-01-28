from dotenv import load_dotenv
load_dotenv()

from liagents.tools.registry import ToolRegistry
from liagents.tools.builtin.calculator import CalculatorTool
from liagents.agents.react_agent import ReActAgent
from liagents.core.client import Client


def test_calculator_tool():
    """测试自定义计算器工具"""

    # 创建包含计算器的注册表
    registry = ToolRegistry()
    python_calculator = CalculatorTool()
    registry.register_tool(python_calculator)

    print("🧪 测试自定义计算器工具\n")

    # 简单测试用例
    test_cases = [
        "2 + 3",           # 基本加法
        "10 - 4",          # 基本减法
        "5 * 6",           # 基本乘法
        "15 / 3",          # 基本除法
        "sqrt(16)",        # 平方根
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("python_calculator", {"expression": expression})
        print(f"结果: {result}\n")


def test_with_simple_agent():
    """测试与SimpleAgent的集成"""

    # 创建LLM客户端
    client = Client()
    llm = ReActAgent(name="ReActAgent", client=client)

    # 创建包含计算器的注册表
    registry = ToolRegistry()
    python_calculator = CalculatorTool()
    registry.register_tool(python_calculator)

    print("与 ReActAgent 集成测试:")

    # 模拟SimpleAgent使用工具的场景
    user_question = "请帮我计算 sqrt(16) + 2 * 3"

    print(f"用户问题: {user_question}")

    # 使用工具计算
    calc_result = registry.execute_tool("python_calculator", {"expression": "sqrt(16) + 2 * 3"})
    print(f"计算结果: {calc_result}")

    # 构建最终回答
    final_messages = [
        {"role": "user", "content": f"计算结果是 {calc_result}，请用自然语言回答用户的问题:{user_question}"}
    ]

    print("\nReActAgent 的回答:")
    response = llm.run(final_messages[0]["content"])
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    test_calculator_tool()
    test_with_simple_agent()
