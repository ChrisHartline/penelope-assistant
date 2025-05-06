# perplexity.py
import logging
import traceback
import time
import requests
from typing import Dict, Any, Optional

from config import ConfigManager

logger = logging.getLogger("penelope.perplexity")

def query_perplexity(query: str, config_manager: Optional[ConfigManager] = None) -> str:
    """Query Perplexity for real-time search and knowledge retrieval."""
    try:
        logger.info(f"Querying Perplexity for: {query}")
        
        # Get API key
        perplexity_api_key = ""
        if config_manager:
            perplexity_api_key = config_manager.get_api_key("perplexity")
        
        if not perplexity_api_key:
            return "Error: Perplexity API key not set. Please configure the API key."
            
        # Set up request
        url = "https://api.perplexity.ai/v1/query"
        headers = {"Authorization": f"Bearer {perplexity_api_key}"}
        payload = {"query": query}
        
        # Make request
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the relevant information from the response
            answer = result.get("text", "No answer found.")
            sources = result.get("references", [])
            
            # Format sources if available
            sources_text = ""
            if sources:
                sources_text = "\n\n📚 Sources:\n"
                for i, source in enumerate(sources, 1):
                    title = source.get("title", "Untitled")
                    url = source.get("url", "No URL")
                    sources_text += f"{i}. {title}: {url}\n"
            
            # Log the search result to a file
            search_log_file = "search_log.txt"
            if config_manager:
                search_log_file = config_manager.get_file_path("search_log_file") or search_log_file
                
            with open(search_log_file, "a", encoding="utf-8") as log:
                log.write(f"QUERY: {query}\n")
                log.write(f"RESULT: {answer[:300]}...\n")
                log.write("-" * 80 + "\n")
                
            formatted_result = f"🔍 Perplexity Search Result:\n\n{answer}{sources_text}"
            logger.info(f"Search completed. Result length: {len(formatted_result)} chars")
            return formatted_result
        else:
            error_msg = f"Error querying Perplexity: {response.status_code}, {response.text}"
            logger.error(error_msg)
            return f"I tried to search for information, but encountered an error: {response.status_code}"
            
    except Exception as e:
        error_msg = f"Perplexity search error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"I tried to search for up-to-date information, but ran into a technical issue."