# arxiv_tools.py
import logging
import io
import time
import json
import traceback
from typing import Dict, Any, Optional, List, Tuple

import arxiv
import requests
from PyPDF2 import PdfReader

from config import ConfigManager
from knowledge_base import KnowledgeBase

logger = logging.getLogger("penelope.arxiv")

def search_arxiv(query: str, max_results: int = 5, config_manager: ConfigManager = None) -> str:
    """Search arXiv for papers matching the query."""
    try:
        logger.info(f"Searching arXiv for: {query} (max results: {max_results})")
        
        # Create search client
        client = arxiv.Client()
        
        # Create search query with Bitcoin/blockchain focus if not specified
        if not any(term in query.lower() for term in ["bitcoin", "blockchain", "crypto"]):
            query = f"{query} AND (bitcoin OR blockchain OR cryptocurrency)"
        
        # Perform search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Get results
        results = list(client.results(search))
        
        if not results:
            return "No papers found matching your query."
        
        # Format results and save paper details
        formatted_results = []
        for i, paper in enumerate(results):
            # Extract paper ID for summarization
            paper_id = paper.entry_id.split('/')[-1]
            
            # Save paper details for future reference
            authors_list = [author.name for author in paper.authors]
            save_paper_details(
                paper_id=paper_id,
                title=paper.title,
                authors=", ".join(authors_list),
                url=paper.pdf_url,
                summary=paper.summary[:500],  # Store a partial summary
                config_manager=config_manager
            )
            
            # Format each paper
            paper_info = (
                f"📄 Paper #{i+1}: {paper.title}\n"
                f"🆔 ID: {paper_id}\n"
                f"👥 Authors: {', '.join(authors_list)}\n"
                f"📅 Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"🔍 Categories: {', '.join(paper.categories)}\n"
                f"🔗 PDF: {paper.pdf_url}\n"
                f"📝 Abstract: {paper.summary[:300]}...\n"
            )
            formatted_results.append(paper_info)
        
        # Combine and return
        return f"Found {len(results)} papers matching '{query}':\n\n" + "\n\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Error searching arXiv: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return f"Error searching arXiv: {str(e)}"

def download_and_extract_text(pdf_url: str) -> str:
    """Download a PDF and extract its text content."""
    try:
        # Download PDF
        response = requests.get(pdf_url, timeout=30)
        if response.status_code != 200:
            return f"Failed to download PDF: HTTP {response.status_code}"
        
        # Convert to BytesIO
        pdf_data = io.BytesIO(response.content)
        
        # Extract text
        reader = PdfReader(pdf_data)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    except Exception as e:
        error_msg = f"Error downloading/extracting PDF: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error processing PDF: {str(e)}"

def save_paper_details(
    paper_id: str, 
    title: str, 
    authors: str, 
    url: Optional[str] = None, 
    summary: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None
) -> bool:
    """Store details about a paper for future reference"""
    try:
        # Get paper details file path
        if config_manager:
            paper_details_file = config_manager.get_file_path("paper_details_file")
        else:
            paper_details_file = "paper_details.json"
            
        # Load existing paper details
        paper_details = {}
        try:
            with open(paper_details_file, "r", encoding="utf-8") as f:
                paper_details = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            paper_details = {}
        
        # Add new paper details
        paper_details[paper_id] = {
            "id": paper_id,
            "title": title,
            "authors": authors,
            "url": url,
            "summary": summary,
            "last_accessed": time.time()
        }
        
        # Save updated details
        with open(paper_details_file, "w", encoding="utf-8") as f:
            json.dump(paper_details, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved details for paper: {title} (ID: {paper_id})")
        return True
        
    except Exception as e:
        logger.error(f"Error saving paper details: {e}", exc_info=True)
        return False

def get_paper_details(config_manager: Optional[ConfigManager] = None) -> Dict[str, Any]:
    """Get details of all tracked papers"""
    try:
        # Get paper details file path
        if config_manager:
            paper_details_file = config_manager.get_file_path("paper_details_file")
        else:
            paper_details_file = "paper_details.json"
            
        try:
            with open(paper_details_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
            
    except Exception as e:
        logger.error(f"Error getting paper details: {e}", exc_info=True)
        return {}

def summarize_arxiv_paper(
    paper_id: str, 
    add_to_kb: bool = False,
    config_manager: Optional[ConfigManager] = None, 
    knowledge_base: Optional[KnowledgeBase] = None,
    claude_model: Optional[Any] = None
) -> str:
    """
    Summarize a paper by its arXiv ID using Claude.
    
    Args:
        paper_id: The arXiv paper ID
        add_to_kb: Whether to add the summary to the knowledge base
        config_manager: Optional ConfigManager instance
        knowledge_base: Optional KnowledgeBase instance
        claude_model: Optional Claude model instance for summarization
    """
    # Load paper details
    if config_manager:
        paper_details_file = config_manager.get_file_path("paper_details_file")
    else:
        paper_details_file = "paper_details.json"
    try:
        with open(paper_details_file, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No paper details found."

    paper = papers.get(paper_id)
    if not paper:
        return f"No details found for paper ID {paper_id}."

    # Compose the prompt for Claude
    prompt = (
        f"Please summarize the following research paper:\n\n"
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Abstract: {paper.get('summary', 'No abstract available.')}\n"
        f"URL: {paper.get('url', 'N/A')}\n"
    )

    if claude_model is None:
        return "Claude model is not available for summarization."

    # Get summary from Claude
    summary = claude_model.get_response(prompt)
    
    # Add to knowledge base if requested
    if add_to_kb and knowledge_base:
        try:
            # Create a document with the paper's content
            doc_content = f"Title: {paper['title']}\nAuthors: {paper['authors']}\n\nSummary:\n{summary}"
            success = knowledge_base.add_document(
                content=doc_content,
                metadata={
                    "source": "arxiv",
                    "paper_id": paper_id,
                    "title": paper['title'],
                    "authors": paper['authors']
                }
            )
            if success:
                summary += "\n\n✅ Added to knowledge base."
            else:
                summary += "\n\n❌ Failed to add to knowledge base."
        except Exception as e:
            logger.error(f"Error adding paper to knowledge base: {e}", exc_info=True)
            summary += f"\n\n❌ Error adding to knowledge base: {str(e)}"
    
    return summary