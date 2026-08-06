# """
# web_search_service.py
#
# Service responsible for performing
# web searches using Tavily.
# """
#
# from __future__ import annotations
#
# import os
#
# from tavily import TavilyClient
#
# from utils.logger import get_logger
#
# logger = get_logger(__name__)
#
#
# class WebSearchService:
#     """
#     Performs internet searches.
#     """
#
#     def __init__(self):
#
#         self.client = TavilyClient(
#             api_key=os.getenv("TAVILY_API_KEY")
#         )
#
#     def search(
#         self,
#         query: str,
#         max_results: int = 5,
#     ) -> list[dict]:
#
#         logger.info(
#             "Searching web: %s",
#             query,
#         )
#
#         response = self.client.search(
#
#             query=query,
#
#             max_results=max_results,
#
#         )
#
#         return response.get(
#             "results",
#             [],
#         )

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults

from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchService:
    """
    Performs internet searches using DuckDuckGo.
    """

    def __init__(self):

        self.search_tool = DuckDuckGoSearchResults(
            output_format="list",
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """
        Search the web.

        Returns a normalized list of dictionaries.
        """

        logger.info(
            "Searching DuckDuckGo: %s",
            query,
        )

        try:

            results = self.search_tool.invoke(query)

            if not isinstance(results, list):
                return []

            return results[:max_results]

        except Exception:

            logger.exception(
                "DuckDuckGo search failed."
            )

            return []