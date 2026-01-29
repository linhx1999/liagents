"""PlanSolveAgent 示例 - 展示基于规划的任务执行模式"""

from dotenv import load_dotenv

load_dotenv()

from liagents.agents.plan_solve_agent import PlanSolveAgent


def test_plan_solve_todo_tracking():
    """测试 PlanSolveAgent 的待办列表追踪功能"""
    print("PlanSolveAgent 待办列表追踪测试")

    agent = PlanSolveAgent(debug=True)

    print("\n已注册工具:", agent.list_tools())

    # 复杂的投资回报分析任务
    task = (
        "请帮我完成一个投资回报分析项目：假设初始投资100000元，年化收益率8%，计算10年后的复利终值；计算该投资在5年、10年、15年、20年后的累计收益；"
    )

    print(f"\n任务: {task}")
    print("\n执行过程:")
    response = agent.run(task)
    print(f"\n最终回答:\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    test_plan_solve_todo_tracking()
