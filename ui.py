# ui.py
import logging
import gradio as gr
import os
from typing import List, Tuple, Dict, Any, Optional

from config import ConfigManager
from models import ClaudeModel
from memory import PenelopeMemory
from knowledge_base import KnowledgeBase
from arxiv_tools import search_arxiv, summarize_arxiv_paper
from perplexity import query_perplexity
# Removed unused imports:
# from arxiv_tools import get_paper_details
from speech import generate_speech

logger = logging.getLogger("penelope.ui")

class PenelopeUI:
    def __init__(self, config_manager: ConfigManager, memory: PenelopeMemory, knowledge_base: KnowledgeBase):
        self.config_manager = config_manager
        self.memory = memory
        self.knowledge_base = knowledge_base
        self.theme = self._create_theme()
        
        logger.info("Initializing Penelope UI")
    
    def _create_theme(self) -> gr.Theme:
        """Create a custom theme for the UI"""
        return gr.themes.Monochrome(
            primary_hue="orange",
            secondary_hue="fuchsia",
        )
    
    def build_ui(self) -> gr.Blocks:
        """Build and return the Gradio UI"""
        with gr.Blocks(theme=self.theme) as demo:
            # Header with logo and title
            with gr.Blocks(css="background: none; border: none;"):
                gr.HTML("""
                <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 0.5em;">
                    <div style="font-size: 3em; margin-right: 0.3em;">🪙</div>
                    <div>
                        <h1 style="margin: 0; font-size: 2.5em; font-weight: 700; background: linear-gradient(90deg, #FF9900, #FFD700); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Penelope</h1>
                        <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">Your Bitcoin & AI Research Assistant</p>
                    </div>
                    <div style="font-size: 3em; margin-left: 0.3em;">⚡</div>
                </div>
                """)
            
            with gr.Tabs():
                # Chat Tab
                with gr.Tab("Chat"):
                    # Chat area
                    chatbot = gr.Chatbot(
                        height="65vh", 
                        show_label=False,
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=penelope"),
                        type="messages"  # Use the new message format
                    )
                    
                    # Status indicator for loading
                    with gr.Row(visible=False) as loading_indicator:
                        gr.HTML("""
                        <div style="display: flex; align-items: center; justify-content: center; margin: 0.5em;">
                            <div class="loading-spinner"></div>
                            <p style="margin-left: 0.5em;">Penelope is thinking...</p>
                        </div>
                        <style>
                            .loading-spinner {
                                width: 20px;
                                height: 20px;
                                border: 3px solid rgba(255, 153, 0, 0.3);
                                border-radius: 50%;
                                border-top-color: #FF9900;
                                animation: spin 1s ease-in-out infinite;
                            }
                            @keyframes spin {
                                to { transform: rotate(360deg); }
                            }
                        </style>
                        """)
                    
                    # Input area with adjusted layout
                    with gr.Row():
                        with gr.Column(scale=8):
                            msg = gr.Textbox(
                                placeholder="Ask Penelope about Bitcoin, AI, or upload research papers...",
                                lines=2,
                                show_label=False,
                                container=True
                            )
                        with gr.Column(scale=1):
                            send_btn = gr.Button(
                                "Send", 
                                scale=1, 
                                size="lg", 
                                icon="paper-plane",
                                variant="primary"
                            )
                    
                    # Audio area with improved styling
                    with gr.Group(visible=False) as audio_group:
                        audio_output = gr.Audio(label="Audio Response")
                        speak_btn = gr.Button("🔊 Listen to Response", size="sm", variant="secondary")
                    
                    # File upload for knowledge base
                    with gr.Accordion("Add to Knowledge Base", open=False):
                        with gr.Row():
                            kb_file = gr.File(label="Upload Document")
                            kb_status = gr.Textbox(label="Upload Status", interactive=False)
                            kb_upload_btn = gr.Button("📤 Add to Knowledge Base", variant="primary", size="sm")
                    
                    # Debug info
                    with gr.Accordion("Debug Info", open=False):
                        debug_info = gr.Textbox(label="Search Debug", lines=5, interactive=False)
                        
                        def update_debug():
                            try:
                                search_log_file = self.config_manager.get_file_path("search_log_file") or "search_log.txt"
                                if os.path.exists(search_log_file):
                                    with open(search_log_file, "r", encoding="utf-8") as f:
                                        return f.read()
                                return "No search log found."
                            except Exception as e:
                                return f"Error reading search log: {e}"
                        
                        refresh_btn = gr.Button("Refresh Debug Info", size="sm")
                        refresh_btn.click(update_debug, inputs=[], outputs=[debug_info])
                    
                    # Help text
                    with gr.Accordion("Help", open=False):
                        gr.Markdown("""
                        ### Special Commands:
                        - Start with "search arxiv: your query" to search for papers
                        - Start with "summarize arxiv: paper_id" to get a paper summary and add to knowledge base
                        - Start with "add to kb: paper_id" to explicitly add a previously retrieved paper to knowledge base
                        - Start with "search kb: your query" to directly search your knowledge base
                        - Start with "check kb" to get statistics about your knowledge base (use "check kb details" for more info)
                        - Use the upload feature to add your own documents to the knowledge base
                        """)
                    
                    # Function to show audio group when there's a response
                    def show_audio_controls(history):
                        if history and len(history) > 0:
                            return gr.Group.update(visible=True)
                        return gr.Group.update(visible=False)
                        
                    # Function to show loading indicator
                    def show_loading():
                        return gr.Row.update(visible=True)
                        
                    # Function to hide loading indicator
                    def hide_loading():
                        return gr.Row.update(visible=False)
                    
                    # Clear chat function
                    def clear_chat():
                        self.memory.reset()
                        logger.info("Chat cleared and memory reset")
                        return [], ""
                    
                    # Upload document function
                    def upload_document(file):
                        if not file:
                            return "Please select a file to upload."
                        
                        try:
                            file_path = file.name
                            file_type = file_path.split('.')[-1].lower()
                            
                            # Check if file type is supported
                            if file_type not in ['pdf', 'txt', 'docx']:
                                return f"Unsupported file type: {file_type}. Please upload PDF, TXT, or DOCX files."
                            
                            # Process document
                            success, message = self.knowledge_base.process_document(file_path, file_type)
                            
                            if success:
                                return f"✅ {message}"
                            else:
                                return f"❌ {message}"
                        except Exception as e:
                            logger.error(f"Error uploading document: {e}", exc_info=True)
                            return f"❌ Error uploading document: {str(e)}"
                    
                    # Chat function
                    def chat(message, history):
                        if not message.strip():
                            return history, ""
                        
                        try:
                            # Check for special commands
                            if message.lower().startswith("search arxiv:"):
                                query = message[13:].strip()
                                result = search_arxiv(query, config_manager=self.config_manager)
                                new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
                                return new_history, ""
                                
                            elif message.lower().startswith("summarize arxiv:"):
                                paper_id = message[15:].strip()
                                result = summarize_arxiv_paper(
                                    paper_id, 
                                    add_to_kb=True, 
                                    config_manager=self.config_manager,
                                    knowledge_base=self.knowledge_base,
                                    claude_model=self.memory.claude_model
                                )
                                new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
                                return new_history, ""
                                
                            elif message.lower().startswith("search kb:"):
                                query = message[10:].strip()
                                kb_results = self.knowledge_base.search_relevant_context(query, top_k=5)
                                result = f"📚 Knowledge Base Search Results for '{query}':\n\n{kb_results}"
                                new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
                                return new_history, ""
                                
                            elif message.lower().startswith("check kb"):
                                detailed = "detail" in message.lower() or "full" in message.lower()
                                result = self._get_kb_info(detailed)
                                new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
                                return new_history, ""
                            
                            elif message.lower().startswith("perplexity:"):
                                query = message[11:].strip()
                                result = query_perplexity(query, config_manager=self.config_manager)
                                new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
                                return new_history, ""
                            
                            # For general questions, first try to get an answer from the knowledge base
                            kb_results = self.knowledge_base.search_relevant_context(message, top_k=3)
                            
                            if kb_results and len(kb_results.strip()) > 0:
                                # If we have relevant knowledge base content, use Claude to synthesize an answer
                                prompt = f"""Based on the following context from our knowledge base, please answer the user's question.
                                If the context doesn't fully answer the question, you can use your general knowledge to supplement the answer.

                                User's question: {message}

                                Relevant context from knowledge base:
                                {kb_results}

                                Please provide a comprehensive answer that combines the knowledge base information with your general knowledge if needed.
                                """
                                response = self.memory.claude_model.get_response(prompt)
                            else:
                                # If no relevant knowledge base content, use Perplexity for general questions
                                response = query_perplexity(message, config_manager=self.config_manager)
                            
                            new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
                            return new_history, ""
                            
                        except Exception as e:
                            logger.error(f"Error in chat: {e}", exc_info=True)
                            error_message = f"I encountered an error: {str(e)}"
                            new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_message}]
                            return new_history, ""
                    
                    # Speak function
                    def speak_last_response(history):
                        try:
                            if not history or len(history) == 0:
                                return None
                            
                            # Get the last assistant response
                            last_response = None
                            for user_msg, assistant_msg in reversed(history):
                                if assistant_msg:
                                    last_response = assistant_msg
                                    break
                                    
                            if not last_response:
                                return None
                                
                            # Generate speech
                            audio_file = generate_speech(last_response, self.config_manager)
                            return audio_file
                            
                        except Exception as e:
                            logger.error(f"Error generating speech: {e}", exc_info=True)
                            return None
                    
                    # Connect UI components
                    send_btn.click(
                        show_loading,
                        None,
                        loading_indicator,
                        queue=False
                    ).then(
                        chat,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    ).then(
                        hide_loading,
                        None,
                        loading_indicator
                    ).then(
                        show_audio_controls,
                        chatbot,
                        audio_group
                    )
                    
                    msg.submit(
                        show_loading,
                        None,
                        loading_indicator,
                        queue=False
                    ).then(
                        chat,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    ).then(
                        hide_loading,
                        None,
                        loading_indicator
                    ).then(
                        show_audio_controls,
                        chatbot,
                        audio_group
                    )
                    
                    speak_btn.click(
                        speak_last_response,
                        inputs=[chatbot],
                        outputs=[audio_output]
                    )
                    
                    kb_upload_btn.click(
                        upload_document,
                        inputs=[kb_file],
                        outputs=[kb_status]
                    )
                    
                    # Clear button
                    clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
                    clear_btn.click(
                        clear_chat,
                        None,
                        [chatbot, msg]
                    )
                
                # arXiv Tools Tab
                with gr.Tab("arXiv Research"):
                    self._create_arxiv_tab()
                
                # Knowledge Base Tab
                with gr.Tab("Knowledge Base"):
                    self._create_kb_tab()
            
            return demo
    
    def _create_arxiv_tab(self):
        """Create the arXiv Research tab"""
        gr.Markdown("## 📚 Bitcoin & Blockchain Research")
        
        with gr.Row():
            with gr.Column(scale=3):
                arxiv_query = gr.Textbox(
                    placeholder="Enter search terms (e.g., bitcoin lightning network)",
                    label="Search Query",
                    show_label=True
                )
            with gr.Column(scale=1):
                max_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Max Results"
                )
                arxiv_search_btn = gr.Button("🔍 Search arXiv", variant="primary")
        
        with gr.Row():
            arxiv_results_text = gr.Textbox(
                label="Search Results",
                lines=12,
                show_label=True
            )
        
        gr.Markdown("## 📄 Summarize Research Paper")
        
        with gr.Row():
            with gr.Column(scale=3):
                paper_id = gr.Textbox(
                    placeholder="Enter paper ID (e.g., 2201.12345)",
                    label="Paper ID",
                    show_label=True
                )
            with gr.Column(scale=1):
                summarize_btn = gr.Button("📝 Summarize Paper", variant="primary")
        
        with gr.Row():
            summary_text = gr.Textbox(
                label="Paper Summary",
                lines=12,
                show_label=True
            )
        
        # Connect arXiv tool buttons
        arxiv_search_btn.click(
            self._search_arxiv_wrapper,
            inputs=[arxiv_query, max_results],
            outputs=arxiv_results_text
        )
        
        summarize_btn.click(
            self._summarize_arxiv_wrapper,
            inputs=[paper_id],
            outputs=summary_text
        )
    
    def _create_kb_tab(self):
        """Create the Knowledge Base tab"""
        gr.Markdown("## 📚 Knowledge Base Management")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔍 Search Your Knowledge")
                kb_query = gr.Textbox(
                    placeholder="Enter a query to search the knowledge base",
                    label="Search Query",
                    show_label=True
                )
                kb_search_btn = gr.Button("🔍 Search Knowledge Base", variant="primary")
                kb_results = gr.Textbox(
                    label="Search Results",
                    lines=10,
                    show_label=True
                )
                
            with gr.Column():
                gr.Markdown("### 📤 Upload Document")
                
                kb_file_upload = gr.File(label="Select Document")
                kb_upload_status = gr.Textbox(label="Upload Status", interactive=False)
                kb_upload_button = gr.Button("📤 Upload to Knowledge Base", variant="primary", size="sm")
        
        gr.Markdown("### 📊 Knowledge Base Statistics")
        
        kb_stats = gr.Textbox(
            label="Knowledge Base Info", 
            lines=10
        )
        kb_refresh_btn = gr.Button("🔄 Refresh Statistics", variant="secondary", size="sm")
        
    def _get_kb_info(self, detailed: bool = False) -> str:
        """Get information about the knowledge base contents"""
        try:
            stats = self.knowledge_base.get_stats()
            
            if not detailed:
                return (
                    f"📚 Knowledge Base Statistics:\n"
                    f"Total Documents: {stats['total_documents']}\n"
                    f"Total Chunks: {stats['total_chunks']}\n"
                    f"Total Tokens: {stats['total_tokens']}\n"
                    f"Last Updated: {stats['last_updated']}"
                )
            
            # Detailed information
            details = ["📚 Knowledge Base Details:"]
            
            # Document types
            details.append("\n📄 Document Types:")
            for doc_type, count in stats.get('document_types', {}).items():
                details.append(f"- {doc_type}: {count} documents")
            
            # Sources
            details.append("\n🔍 Sources:")
            for source, count in stats.get('sources', {}).items():
                details.append(f"- {source}: {count} documents")
            
            # Recent documents
            details.append("\n📅 Recent Documents:")
            for doc in stats.get('recent_documents', [])[:5]:
                details.append(f"- {doc['title']} ({doc['date_added']})")
            
            return "\n".join(details)
            
        except Exception as e:
            logger.error(f"Error getting KB info: {e}", exc_info=True)
            return f"Error getting knowledge base information: {str(e)}"
    
    def _search_arxiv_wrapper(self, query: str, max_results: int) -> str:
        """Wrapper for arXiv search functionality"""
        try:
            if not query.strip():
                return "Please enter a search query."
            
            result = search_arxiv(
                query=query,
                max_results=max_results,
                config_manager=self.config_manager
            )
            return result
            
        except Exception as e:
            logger.error(f"Error in arXiv search: {e}", exc_info=True)
            return f"Error searching arXiv: {str(e)}"

    def _summarize_arxiv_wrapper(self, paper_id: str) -> str:
        """Wrapper for arXiv paper summarization"""
        try:
            if not paper_id.strip():
                return "Please enter a paper ID."
            
            result = summarize_arxiv_paper(
                paper_id=paper_id,
                add_to_kb=True,
                config_manager=self.config_manager,
                knowledge_base=self.knowledge_base,
                claude_model=self.memory.claude_model
            )
            return result
            
        except Exception as e:
            logger.error(f"Error summarizing paper: {e}", exc_info=True)
            return f"Error summarizing paper: {str(e)}"
    
if __name__ == "__main__":
    # Initialize components
    config_manager = ConfigManager()
    claude_model = ClaudeModel(config_manager)  # Create Claude model first
    memory = PenelopeMemory(config_manager, claude_model)  # Pass claude_model to memory
    knowledge_base = KnowledgeBase(config_manager)
    
    # Create and launch UI
    ui = PenelopeUI(config_manager, memory, knowledge_base)
    demo = ui.build_ui()
    
    # Launch the interface
    demo.launch(
        server_name="127.0.0.1",  # Use localhost
        server_port=7860,         # Default Gradio port
        share=True,               # Creates a public URL
        inbrowser=True           # Automatically open in browser
    )
     