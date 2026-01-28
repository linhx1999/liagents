"""Tavily搜索工具 - 使用 @tool 装饰器实现"""

import os
from typing import Annotated

from ..base import tool

_tavily_client = None


def _get_tavily_client():
    """获取或初始化Tavily客户端"""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            try:
                from tavily import TavilyClient

                _tavily_client = TavilyClient(api_key=api_key)
            except ImportError:
                pass
    return _tavily_client


@tool
def tavily_search(
    query: Annotated[str, "搜索查询关键词"],
    max_results: Annotated[int, "返回结果的最大数量"] = 5,
    include_answer: Annotated[bool, "是否包含AI生成的答案"] = True,
) -> str:
    """Tavily AI搜索工具。基于Tavily搜索引擎进行网络搜索，返回结构化的搜索结果。

    例如：搜索最新的AI技术新闻、查询某个问题的答案等。
    """
    if not query:
        return "错误：搜索查询不能为空"

    client = _get_tavily_client()

    if client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "错误：未设置 TAVILY_API_KEY 环境变量。请前往 https://tavily.com/ 获取API密钥"
        return "错误：Tavily客户端初始化失败，请检查API密钥是否正确"

    print(f"[Tool: tavily_search] 正在搜索: {query}")

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=include_answer,
        )

        result_parts = []

        if include_answer and response.get("answer"):
            result_parts.append(f"AI答案: {response['answer']}")

        results = response.get("results", [])
        if results:
            result_parts.append(f"相关结果 (共{len(results)}条):\n")
            for i, item in enumerate(results, 1):
                title = item.get("title", "无标题")
                content = item.get("content", "")
                url = item.get("url", "")
                result_parts.append(f"[{i}] {title}")
                result_parts.append(
                    f"    {content[:1000]}..."
                    if len(content) > 1000
                    else f"    {content}"
                )
                if url:
                    result_parts.append(f"    来源: {url}")
                result_parts.append("")

        final_result = "\n".join(result_parts)
        print(f"[Tool: tavily_search] 搜索完成，返回 {len(results)} 条结果")
        return final_result if final_result else "未找到相关结果"

    except Exception as e:
        error_msg = f"搜索失败: {str(e)}"
        print(f"[Tool: tavily_search] {error_msg}")
        return error_msg
