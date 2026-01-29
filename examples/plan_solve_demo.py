"""PlanSolveAgent 示例 - 展示基于规划的任务执行模式"""

from dotenv import load_dotenv

load_dotenv()

from liagents.agents.plan_solve_agent import PlanSolveAgent
from liagents.tools.builtin.calculator import python_calculator


def test_plan_solve_todo_tracking():
    """测试 PlanSolveAgent 的待办列表追踪功能"""
    print("\n" + "=" * 60)
    print("PlanSolveAgent 待办列表追踪测试")

    agent = PlanSolveAgent()

    # 添加计算工具用于后续计算
    agent.add_tool(python_calculator)

    print("\n已注册工具:", agent.list_tools())

    # 复杂的投资回报分析任务
    task = "帮我写一个贪吃蛇游戏，用 Python，并写好测试用例。"

    print(f"\n任务: {task}")
    print("\n执行过程:")
    response = agent.run(task)
    print(f"\n最终回答:\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    test_plan_solve_todo_tracking()
