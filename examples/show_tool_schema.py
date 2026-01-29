"""展示工具的 OpenAI function calling schema 格式"""

import json
from liagents.tools.builtin import (
    python_calculator,
    tavily_search,
    write_todos,
)

# 收集所有工具
tools = [python_calculator, tavily_search, write_todos]


def main():
    # 展示所有工具的基本信息
    for tool in tools:
        print("=" * 60)
        print(f"工具名称: {tool.name}")
        print(f"工具描述:\n{tool.description}")

        print("-" * 40)
        print("工具参数定义（内部格式）")
        for param in tool.get_parameters():
            print(f"{param}")

        print("-" * 40)
        print("OpenAI Function Calling Schema 格式")
        schema = tool.to_schema()
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        print()

    # 展示批量请求示例
    print("=" * 60)
    print("实际请求示例")
    print("如果使用 OpenAI 原生 function calling，请求格式如下：")
    request_example = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "帮我计算 sqrt(16) + 2 * 3"}],
        "tools": [tool.to_schema()["function"] for tool in tools],
        "tool_choice": "auto",
    }
    print(json.dumps(request_example, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
