# memory.py
import json
import logging
import uuid
import time
from typing import List, Dict, Any, Optional

from langchain.memory import ConversationSummaryMemory

from config import ConfigManager
from models import ClaudeModel

logger = logging.getLogger("penelope.memory")

class PenelopeMemory:
    def __init__(self, config_manager: ConfigManager, claude_model: ClaudeModel):
        self.config_manager = config_manager
        self.claude_model = claude_model
        self.thread_id = str(uuid.uuid4())
        self.messages = []
        
        # Get memory file path
        self.memory_file = config_manager.get_file_path("memory_file")
        if not self.memory_file:
            self.memory_file = "conversation_history.json"
        
        # Get paper details file path
        self.paper_details_file = config_manager.get_file_path("paper_details_file")
        if not self.paper_details_file:
            self.paper_details_file = "paper_details.json"
        
        # Initialize summary memory
        self.summary_memory = ConversationSummaryMemory(
            llm=claude_model.model,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Load existing messages
        self._load_messages()
        
        logger.info(f"Memory system initialized with thread ID: {self.thread_id}")
    
    def _load_messages(self) -> None:
        """Load messages from file"""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
                logger.info(f"Loaded {len(self.messages)} messages from {self.memory_file}")
                
                # Also add to summary memory
                for msg in self.messages:
                    if msg["role"] == "user":
                        self.summary_memory.chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        self.summary_memory.chat_memory.add_ai_message(msg["content"])
        except (FileNotFoundError, json.JSONDecodeError):
            self.messages = []
            logger.info(f"No previous messages found or error loading from {self.memory_file}")
    
    def get_paper_details_context(self) -> str:
        """Get formatted paper details for memory context"""
        try:
            paper_details = self._get_paper_details()
            if not paper_details:
                return "No papers have been discussed yet."
                
            # Format paper details
            formatted_papers = []
            for paper_id, paper in paper_details.items():
                formatted_papers.append(
                    f"Paper ID: {paper_id}\n"
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"URL: {paper.get('url', 'N/A')}"
                )
            
            return "Papers discussed in this conversation:\n\n" + "\n\n".join(formatted_papers)
        except Exception as e:
            logger.error(f"Error getting paper details: {str(e)}", exc_info=True)
            return "Error retrieving paper details."
    
    def _get_paper_details(self) -> Dict[str, Any]:
        """Get details of all tracked papers"""
        try:
            try:
                with open(self.paper_details_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        except Exception as e:
            logger.error(f"Error loading paper details: {str(e)}", exc_info=True)
            return {}
    
    def add_message(self, message: str, is_human: bool = True) -> None:
        """Add a message to the memory"""
        # Add to conversation history
        if is_human:
            self.summary_memory.chat_memory.add_user_message(message)
            self.messages.append({"role": "user", "content": message})
        else:
            self.summary_memory.chat_memory.add_ai_message(message)
            self.messages.append({"role": "assistant", "content": message})
        
        # Save to file for persistence
        self._save_messages()
    
    def _save_messages(self) -> None:
        """Save messages to file"""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.messages)} messages to {self.memory_file}")
        except Exception as e:
            logger.error(f"Error saving to memory file: {str(e)}", exc_info=True)
    
    def get_response(self, query: str, context: Optional[str] = None) -> str:
        """Get a response using the memory system"""
        try:
            # Check if we need to summarize (every 10 messages)
            if len(self.messages) >= 10 and len(self.messages) % 10 == 0:
                logger.info("Generating conversation summary...")
                # Get summary
                summary = self.summary_memory.load_memory_variables({})
                logger.info(f"Summary generated: {len(str(summary))} chars")
            
            # Get paper details
            papers_context = self.get_paper_details_context()
            
            # Final context combines provided context and papers
            final_context = ""
            if context:
                final_context += context + "\n\n"
            final_context += f"Papers discussed: {papers_context}"
            
            # Get response from Claude
            if len(self.messages) >= 10:
                # Use summarized history
                summary = self.summary_memory.load_memory_variables({})
                history_context = f"Here's a summary of our conversation so far:\n{summary['chat_history']}\n\n"
                final_context = history_context + final_context
                
                # Add the summary message to history for context
                summary_history = [
                    {"role": "user", "content": history_context},
                    {"role": "assistant", "content": "I understand the conversation context and the papers we've discussed. What would you like to know next?"}
                ]
                
                # Use only these messages as history
                response = self.claude_model.get_response(query, history=summary_history, context=final_context)
            else:
                # Use full message history
                response = self.claude_model.get_response(query, history=self.messages, context=final_context)
            
            # Add the messages to memory
            self.add_message(query, is_human=True)
            self.add_message(response, is_human=False)
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I encountered an error: {str(e)}. Please try again."
    
    def reset(self) -> None:
        """Reset the memory"""
        self.messages = []
        self.summary_memory.clear()
        self.thread_id = str(uuid.uuid4())
        self._save_messages()
        logger.info(f"Memory reset. New thread ID: {self.thread_id}")