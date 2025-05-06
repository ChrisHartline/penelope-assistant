# models.py
import logging
from typing import List, Dict, Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import ConfigManager, SYSTEM_PROMPT

logger = logging.getLogger("penelope.models")

class ClaudeModel:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.model_settings = config_manager.get_model_settings()
        
        # Get API key
        anthropic_api_key = config_manager.get_api_key("anthropic")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required for Claude model")
        
        # Initialize Claude
        self.model = ChatAnthropic(
            model=self.model_settings.get("claude", "claude-3-7-sonnet-20250219"),
            temperature=self.model_settings.get("temperature", 0.7),
            anthropic_api_key=anthropic_api_key,
            system=SYSTEM_PROMPT
        )
        
        logger.info(f"Claude model initialized with {self.model_settings.get('claude')}")
    
    def get_response(self, message: str, history: Optional[List[Dict[str, str]]] = None, 
                     context: Optional[str] = None) -> str:
        """Get response from Claude with optional context and history"""
        try:
            # Prepare the messages list
            messages = []
            
            # Add message history if provided
            if history:
                for msg in history[-6:]:  # Last 6 messages for context
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            # Add context to the message if provided
            if context:
                augmented_message = f"{message}\n\n{context}"
            else:
                augmented_message = message
                
            # Add the current message
            messages.append(HumanMessage(content=augmented_message))
            
            logger.info(f"Sending message to Claude (length: {len(augmented_message)} chars)")
            
            # Get response from Claude
            response = self.model.invoke(messages)
            
            # Extract the content
            return response.content
            
        except Exception as e:
            error_msg = f"Error getting Claude response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I encountered an error communicating with my language model: {str(e)}"