import logging
import re
from typing import List, Dict, Union
from langchain_core.tools import Tool

from vector_store.chroma_db import ChromaDB
from utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class KnowledgeBaseTool:
    """
    A LangChain Tool for searching the internal ITC/AMS knowledge base stored in ChromaDB.
    Includes security checks to block system-related queries.
    """
    def __init__(self, vector_db: ChromaDB, document_processor: DocumentProcessor, domain_filters: Dict):
        self.vector_db = vector_db
        self.document_processor = document_processor
        self.domain_filters = domain_filters
        logger.info("KnowledgeBaseTool initialized.")

    def _is_itc_ams_domain(self, url: str) -> bool:
        itc_domains = self.domain_filters.get('itc', [])
        ams_domains = self.domain_filters.get('ams', [])
        
        all_allowed_patterns = itc_domains + ams_domains
        
        for pattern in all_allowed_patterns:
            regex_pattern = re.escape(pattern).replace(r'\*\*', '.*').replace(r'\*', '[^/]*')
            if re.fullmatch(regex_pattern, url):
                return True
        return False

    def _is_system_query(self, query: str) -> bool:
        """Detect queries about system internals"""
        system_keywords = [
            r'\b(?:system|database|vector|chroma|storage|knowledge base|internals?|how (?:you|it) works?)\b',
            r'\b(?:api|implementation|architecture|design|security|privacy)\b',
            r'\b(?:download|export|dump|backup)\b.*\b(?:data|database)\b',
            r'\b(?:hack|exploit|vulnerability|bypass|security|inject)\b',
            r'\b(?:delete|remove|destroy|erase|reset|drop)\b.*\b(?:data|database|system)\b',
            r'\b(?:source code|config|\.env|api key|secret)\b'
        ]
        
        query_lower = query.lower()
        for pattern in system_keywords:
            if re.search(pattern, query_lower):
                return True
        return False

    def _run(self, query: str) -> str:
        # Block any queries trying to access system information
        if self._is_system_query(query):
            return "Access to system information is restricted. Please ask about ITC/AMS topics."
        
        logger.info(f"Searching internal knowledge base for: '{query}'")
        
        try:
            results = self.vector_db.query_documents(query_texts=[query], n_results=25)

            domain_results = [
                doc for doc in results 
                if self._is_itc_ams_domain(doc.get('metadata', {}).get('source', ''))
            ]
            
            if not domain_results:
                return "No relevant information found in our ITC/AMS domain knowledge base for this query. 🔍"

            domain_results.sort(key=lambda x: x.get('distance', float('inf')))
            top_results = domain_results[:10]

            formatted_output = []
            for i, doc in enumerate(top_results):
                source = doc.get('metadata', {}).get('source', 'N/A')
                formatted_output.append(
                    f"--- Retrieved Document {i+1} (Source: {source}) ---\n"
                    f"{doc['text']}\n"
                    f"---"
                )
            
            logger.info(f"KnowledgeBaseTool returned {len(top_results)} relevant chunks for '{query}'.")
            return "\n\n".join(formatted_output)

        except Exception as e:
            logger.error(f"Error querying Knowledge Base for '{query}': {e}", exc_info=True)
            return f"An error occurred while searching the internal knowledge base: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="knowledge_base_search",
            description="Useful for searching the internal knowledge base of ITC and AMS documents. "
                        "This tool can only retrieve information that has been specifically ingested from ITC and AMS official domains. "
                        "Input is a concise search query string related to ITC or AMS topics (e.g., 'AMS master programs', 'ITC admission requirements'). "
                        "Returns highly relevant text snippets (chunks) from the indexed internal documents. "
                        "**Always use this tool first for any questions about ITC or AMS.**",
            func=self._run
        )