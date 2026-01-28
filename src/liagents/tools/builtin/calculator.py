"""计算器工具 - 使用 @tool 装饰器重新实现"""

import ast
import operator
import math
from typing import Annotated

from ..base import tool


# 支持的操作符
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
}

# 支持的函数
FUNCTIONS = {
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node):
    """递归计算AST节点"""
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.BinOp):
        return OPERATORS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    elif isinstance(node, ast.UnaryOp):
        return OPERATORS[type(node.op)](_eval_node(node.operand))
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in FUNCTIONS:
            args = [_eval_node(arg) for arg in node.args]
            return FUNCTIONS[func_name](*args)
        else:
            raise ValueError(f"不支持的函数: {func_name}")
    elif isinstance(node, ast.Name):
        if node.id in FUNCTIONS:
            return FUNCTIONS[node.id]
        else:
            raise ValueError(f"未定义的变量: {node.id}")
    else:
        raise ValueError(f"不支持的表达式类型: {type(node)}")


@tool
def python_calculator(expression: Annotated[str, "要计算的数学表达式"]) -> str:
    """Python计算器工具。执行数学计算，支持基本运算、数学函数等。

    例如：2+3*4, sqrt(16), sin(pi/2)等。
    """
    if not expression:
        return "错误：计算表达式不能为空"

    print(f"[Tool: python_calculator] 正在计算: {expression}")

    try:
        # 解析表达式
        node = ast.parse(expression, mode="eval")
        result = _eval_node(node.body)
        result_str = str(result)
        print(f"[Tool: python_calculator] 计算结果: {result_str}")
        return result_str
    except Exception as e:
        error_msg = f"计算失败: {str(e)}"
        print(f"[Tool: python_calculator] {error_msg}")
        return error_msg
