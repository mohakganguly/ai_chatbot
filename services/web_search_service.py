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

from tavily import TavilyClient

from config import TAVILY_API_KEY
from utils.logger import get_logger


logger = get_logger(__name__)


class WebSearchService:
    """
    Performs internet searches using Tavily.
    """

    def __init__(self):
        self.client = TavilyClient(
            api_key=TAVILY_API_KEY
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """
        Search the web using Tavily.

        Returns a normalized list of dictionaries.
        """

        logger.info(
            "Searching Tavily: %s",
            query,
        )

        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )

            results = response.get(
                "results",
                [],
            )

            normalized_results = []

            for result in results:
                normalized_results.append(
                    {
                        "title": result.get(
                            "title",
                            "",
                        ),
                        "url": result.get(
                            "url",
                            "",
                        ),
                        "content": result.get(
                            "content",
                            "",
                        ),
                    }
                )

            return normalized_results

        except Exception:
            logger.exception(
                "Tavily search failed."
            )

            return []