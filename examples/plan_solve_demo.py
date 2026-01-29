"""PlanSolveAgent 示例 - 展示基于规划的任务执行模式"""

from dotenv import load_dotenv

load_dotenv()

from liagents.agents.plan_solve_agent import PlanSolveAgent
from liagents.tools.builtin.calculator import python_calculator


def test_plan_solve_todo_tracking(task):
    """测试 PlanSolveAgent 的待办列表追踪功能"""
    print("=" * 60)
    print("PlanSolveAgent 待办列表追踪测试")

    agent = PlanSolveAgent(debug=True)

    print("\n已注册工具:", agent.list_tools())

    print(f"\n任务: {task}")
    print("\n执行过程:")
    response = agent.run(task)
    print(f"\n最终回答:\n{response}")


def test_calculator_with_agent(task):
    """测试计算器工具并展示 Agent debug 输出"""
    print("=" * 60)
    print("PlanSolveAgent 带计算器工具测试")

    agent = PlanSolveAgent(debug=True)
    agent.add_tool(python_calculator)

    print("\n已注册工具:", agent.list_tools())

    print(f"\n任务: {task}")
    print("\n执行过程:")
    response = agent.run(task)
    print(f"\n最终回答:\n{response}")


if __name__ == "__main__":
    task = "请帮我完成一个投资回报分析项目：假设初始投资100000元，年化收益率8%，计算10年后的复利终值；计算该投资在5年、10年、15年、20年后的累计收益。"

    test_plan_solve_todo_tracking(task)
    print("\n")
    test_calculator_with_agent(task)
