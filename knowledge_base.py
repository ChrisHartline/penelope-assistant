# knowledge_base.py
import os
import time
import logging
from typing import Tuple, List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import docx2txt
from PyPDF2 import PdfReader

from config import ConfigManager

logger = logging.getLogger("penelope.knowledge_base")

class KnowledgeBase:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.kb_settings = config_manager.get_kb_settings()
        
        # Get the OpenAI API key
        openai_api_key = config_manager.get_api_key("openai")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for knowledge base embeddings")
        
        # Initialize embeddings
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        
        # Initialize ChromaDB
        persist_dir = self.kb_settings.get("persist_dir", "./chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_function,
            collection_name="penelope_knowledge"
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.kb_settings.get("chunk_size", 1000),
            chunk_overlap=self.kb_settings.get("chunk_overlap", 200),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"Knowledge Base initialized at {persist_dir}")
    
    def process_document(self, file_path: str, file_type: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Process document and add to knowledge base"""
        try:
            logger.info(f"Processing document: {file_path}")
            text = self.extract_text(file_path, file_type)
            
            if not text:
                return False, "Failed to extract text from document"
            
            # Create metadata
            if metadata is None:
                metadata = {}
            
            metadata["source"] = file_path
            metadata["date_added"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata["file_type"] = file_type
            
            # Split text into chunks
            docs = self.text_splitter.create_documents(
                texts=[text], 
                metadatas=[metadata]
            )
            
            # Add to vector store
            self.db.add_documents(docs)
            
            # Persist to disk
            try:
                self.db.persist()
                logger.info("Successfully persisted to disk")
            except Exception as persist_error:
                logger.warning(f"Could not persist to disk: {persist_error}")
                # Continue anyway since documents were added to memory
            
            logger.info(f"Added {len(docs)} chunks from {file_path} to knowledge base")
            return True, f"Successfully added {len(docs)} chunks to knowledge base"
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def extract_text(self, file_path: str, file_type: str) -> Optional[str]:
        """Extract text from different file types"""
        try:
            if file_type == "pdf":
                return self.extract_text_from_pdf(file_path)
            elif file_type == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_type == "docx":
                return docx2txt.process(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return None
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}", exc_info=True)
            return None
    
    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            with open(file_path, "rb") as f:
                pdf = PdfReader(f)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}", exc_info=True)
            return None
    
    def search_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Search for relevant context based on query"""
        try:
            logger.info(f"Searching knowledge base for: '{query}'")
            results = self.db.similarity_search_with_relevance_scores(query, k=top_k)
            
            formatted_results = []
            relevance_threshold = self.kb_settings.get("relevance_threshold", 0.2)
            
            for doc, score in results:
                if score > relevance_threshold:
                    source = doc.metadata.get("source", "Unknown source")
                    source_type = doc.metadata.get("file_type", doc.metadata.get("type", ""))
                    title = doc.metadata.get("title", "")
                    
                    # Create a better formatted context
                    if title:
                        header = f"Source: {title} ({source_type})"
                    else:
                        header = f"Source: {os.path.basename(source)} ({source_type})"
                        
                    formatted_content = f"{header}\nRelevance: {score:.2f}\nContent: {doc.page_content}\n"
                    formatted_results.append(formatted_content)
                    logger.info(f"Found relevant document: {header} with score {score:.2f}")
            
            if not formatted_results:
                logger.info("No relevant information found in knowledge base.")
                return "No relevant information found in knowledge base."
            
            context = "\n\n".join(formatted_results)
            logger.info(f"Retrieved {len(formatted_results)} relevant passages from knowledge base")
            return context
            
        except Exception as e:
            error_msg = f"Error searching knowledge base: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return "Error retrieving information from knowledge base."