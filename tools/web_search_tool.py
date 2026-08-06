"""
web_search_tool.py

Web Search Tool.
"""

from __future__ import annotations

from services.web_search_service import (
    WebSearchService,
)

from tools.base import (
    BaseTool,
    ToolContext,
    ToolResult,
)


class WebSearchTool(BaseTool):

    name = "web_search"

    description = (
        "Search the internet for recent information, "
        "news, documentation, APIs, releases and "
        "public knowledge."
    )

    def __init__(self):

        self.service = WebSearchService()

    def invoke(
        self,
        context: ToolContext,
    ) -> ToolResult:

        results = self.service.search(
            context.query
        )

        if not results:

            return ToolResult(

                tool_name=self.name,

                success=False,

                summary="No search results found.",

                error="No results returned.",
            )

        summary = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            summary.append(

                f"""{index}.

 

                Title:
                {result.get("title", "N/A")}

                Snippet:
                {result.get("snippet", result.get("content", ""))}

                URL:
                {result.get("link", result.get("url", "N/A"))}
                """
            )

        return ToolResult(

            tool_name=self.name,

            success=True,

            output=results,

            summary="\n\n".join(summary),
        )


# Title:
# {result.get("title")}
#
# Content:
# {result.get("content")}
#
# Source:
# {result.get("url")}
# """