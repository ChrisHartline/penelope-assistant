# config.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("penelope.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("penelope.config")

# System prompts
BASE_PROMPT = """You are Penelope, a helpful and knowledgeable assistant who is passionate about AI, Bitcoin, cryptocurrency and quantum computing."""

PERSONALITY_TRAITS = """
As an enthusiast, you:
- Speak positively about AI, Bitcoin, cryptocurrency and quantum computing
- Use AI, Bitcoin, cryptocurrency and quantum computing terminology naturally in conversation
- Are up-to-date on AI, Bitcoin, cryptocurrency and quantum computing developments, market trends, and adoption news    
- Can explain complex AI, Bitcoin, cryptocurrency and quantum computing concepts in accessible terms    
- Enjoy discussing topics like AI, Bitcoin, cryptocurrency and quantum computing investing, blockchain technology, and the future of finance    
"""

# Additional prompt components
TOOL_INSTRUCTIONS = """
You have access to the following tools:
1. arXiv Search: Search for research papers on arXiv
2. Paper Summarization: Get AI-generated summaries of papers
3. Knowledge Base: Store and retrieve information from papers and documents
4. Perplexity: Answer general questions about AI, Bitcoin, and related topics
"""

KB_INSTRUCTIONS = """
KNOWLEDGE BASE:
- You can search the knowledge base for relevant information
- Papers and documents are automatically added to the knowledge base
- Use the knowledge base to provide informed answers about stored content
- You can check the knowledge base status and contents
"""

PAPER_INSTRUCTIONS = """
PAPER TRACKING SYSTEM:
- Papers are tracked by their arXiv IDs
- You can search for papers using natural language queries
- Paper summaries are generated using Claude
- Summarized papers are automatically added to the knowledge base
"""

SEARCH_INSTRUCTIONS = """
For research-related questions:
1. First check if the information is in the knowledge base
2. If not found, search arXiv for relevant papers
3. Use Perplexity for general questions not related to papers
4. Always cite sources when providing information from papers
"""

CRITICAL_INSTRUCTIONS = """
CRITICAL INSTRUCTION: When I automatically search for information:
1. Always verify the information is relevant to the query
2. Provide clear citations and sources
3. Explain complex concepts in accessible terms
4. Be honest about limitations or uncertainties
5. Focus on accuracy and helpfulness
"""

# Combine all prompt components
SYSTEM_PROMPT = (
    BASE_PROMPT + PERSONALITY_TRAITS + TOOL_INSTRUCTIONS + 
    KB_INSTRUCTIONS + PAPER_INSTRUCTIONS + SEARCH_INSTRUCTIONS + 
    CRITICAL_INSTRUCTIONS
)

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables"""
        # Default config
        config = {
            "api_keys": {
                "anthropic": "",
                "elevenlabs": "",
                "perplexity": "",
                "openai": ""
            },
            "models": {
                "claude": "claude-3-7-sonnet-20250219",
                "temperature": 0.7
            },
            "knowledge_base": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "relevance_threshold": 0.2,
                "persist_dir": "./chroma_db"
            },
            "files": {
                "memory_file": "conversation_history.json",
                "paper_details_file": "paper_details.json",
                "search_log_file": "search_log.txt"
            }
        }
        
        # Try to load from file
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    loaded_config = json.load(f)
                    self._update_nested_dict(config, loaded_config)
                    logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        # Override with environment variables if present
        for key in config["api_keys"]:
            env_var = f"{key.upper()}_API_KEY"
            if env_var in os.environ:
                config["api_keys"][key] = os.environ[env_var]
                logger.info(f"Loaded {key} API key from environment")
        
        return config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            return False
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a service"""
        return self.config["api_keys"].get(service, "")
    
    def set_api_key(self, service: str, key: str) -> bool:
        """Set API key for a service"""
        if service in self.config["api_keys"]:
            self.config["api_keys"][service] = key
            return self.save_config()
        return False
    
    def get_model_settings(self) -> Dict[str, Any]:
        """Get model settings"""
        return self.config["models"]
    
    def get_kb_settings(self) -> Dict[str, Any]:
        """Get knowledge base settings"""
        return self.config["knowledge_base"]
    
    def get_file_path(self, file_key: str) -> str:
        """Get file path from config"""
        return self.config["files"].get(file_key, "")
    
    def ensure_api_keys(self) -> bool:
        """Ensure all required API keys are set"""
        all_keys_set = True
        for service, key in self.config["api_keys"].items():
            if not key:
                logger.warning(f"API key for {service} is not set")
                all_keys_set = False
        return all_keys_set