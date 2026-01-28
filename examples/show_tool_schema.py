"""展示工具的 OpenAI function calling schema 格式"""

from liagents.tools.builtin.calculator import python_calculator
import json


def main():
    print("=" * 60)
    print("工具基本信息")
    print(f"工具名称: {python_calculator.name}")
    print(f"工具描述:\n{python_calculator.description}")

    print("=" * 60)
    print("工具参数定义（内部格式）")
    for param in python_calculator.get_parameters():
        print(f"{param}")

    print("=" * 60)
    print("OpenAI Function Calling Schema 格式")
    schema = python_calculator.to_schema()
    print(json.dumps(schema, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("实际请求示例")
    print("如果使用 OpenAI 原生 function calling，请求格式如下：")
    request_example = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "帮我计算 sqrt(16) + 2 * 3"}
        ],
        "tools": [schema["function"]],  # 注意：取 schema 中的 function 部分
        "tool_choice": "auto"
    }
    print(json.dumps(request_example, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
