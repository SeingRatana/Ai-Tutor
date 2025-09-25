import os
import requests
import logging
from typing import List, Dict, Union
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

class TavilySearchTool:
    """Tool for performing web searches using the Tavily API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"
        self.headers = {'Content-Type': 'application/json', 'User-Agent': 'ITC-AMS-SearchBot/1.0'}
        logger.info("TavilySearchTool initialized.")

    def _run(self, query: str, max_results: int = 5) -> str:
        payload = {
            "api_key": self.api_key,
            "q": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
            "include_images": False,
            "include_video": False
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for res in data.get('results', []):
                results.append(
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    f"Snippet: {res.get('content', 'No snippet available.')}\n---"
                )
            
            if not results:
                return "No results found from Tavily Search."
            
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error in TavilySearchTool: {e}", exc_info=True)
            return f"Error occurred during Tavily search: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="web_search_tavily",
            description="Useful for general web searches when you need up-to-date information. "
                        "Input should be a concise search query string. "
                        "Returns a list of relevant search results with title, link, and snippet.",
            func=self._run
        )

class BraveSearchTool:
    """Tool for performing web searches using the Brave Search API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key,
            'User-Agent': 'ITC-AMS-SearchBot/1.0'
        }
        logger.info("BraveSearchTool initialized.")

    def _run(self, query: str) -> str:
        params = {'q': query, 'country': 'KH'}
        try:
            response = requests.get(self.endpoint, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for res in data.get('web', {}).get('results', []):
                results.append(
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    f"Snippet: {res.get('description', 'No snippet available.')}\n---"
                )
            
            if not results:
                return "No results found from Brave Search."
            
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error in BraveSearchTool: {e}", exc_info=True)
            return f"Error occurred during Brave search: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="web_search_brave",
            description="Useful for general web searches providing an independent index. "
                        "Input should be a concise search query string. "
                        "Returns a list of relevant search results with title, link, and snippet.",
            func=self._run
        )

class DuckDuckGoSearchTool:
    """Tool for performing general web searches using DuckDuckGo."""
    def __init__(self):
        logger.info("DuckDuckGoSearchTool initialized.")
        self.endpoint = "https://api.duckduckgo.com/?format=json&q="
        self.headers = {'User-Agent': 'ITC-AMS-SearchBot/1.0'}

    def _run(self, query: str) -> str:
        try:
            response = requests.get(f"{self.endpoint}{query}", headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if data.get('AbstractText'):
                results.append(f"Abstract: {data['AbstractText']}\n")
            elif data.get('Abstract'):
                results.append(f"Abstract: {data['Abstract']}\n")
            
            if data.get('Results'):
                for res in data['Results']:
                    results.append(f"Title: {res.get('Text', 'N/A')}\nURL: {res.get('FirstURL', 'N/A')}\n---")
            
            if not results:
                return "No results found from DuckDuckGo Search."

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error in DuckDuckGoSearchTool: {e}", exc_info=True)
            return f"Error occurred during DuckDuckGo search: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="web_search_duckduckgo",
            description="Useful for general web searches, especially for quick, factual answers or when other APIs fail. "
                        "Input is a concise search query string. Returns search results with titles, links, and snippets.",
            func=self._run
        )

class ScraperTool:
    """Tool for scraping content from a given URL."""
    def __init__(self, user_agent: str, document_processor, vector_db):
        self.user_agent = user_agent
        self.document_processor = document_processor
        self.vector_db = vector_db
        logger.info("ScraperTool initialized.")

    def _run(self, url: str) -> str:
        logger.info(f"Attempting to scrape URL: {url}")
        try:
            extracted_text = self.document_processor.scrape_and_extract_content(
                url=url,
                user_agent=self.user_agent,
                max_depth=0
            )
            
            if extracted_text and extracted_text.strip():
                return extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text
            else:
                return f"No meaningful content extracted from {url}."
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}", exc_info=True)
            return f"Failed to scrape content from {url}: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="web_scraper",
            description="Useful for extracting the full text content from a given URL. "
                        "Input is a single URL string. Returns the extracted text content.",
            func=self._run
        )