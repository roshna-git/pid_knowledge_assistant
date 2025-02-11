"""
Main application file with integrated P&ID processing and querying capabilities
"""

import gradio as gr
import logging
import os
from components.nlp_processor import NLPProcessor
from components.knowledge_graph import KnowledgeGraph
from components.vector_store import VectorStore
from components.pid_processor import PIDProcessor
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PIDAssistant:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kg = KnowledgeGraph()
        self.vs = VectorStore()
        self.nlp = NLPProcessor(knowledge_graph=self.kg, vector_store=self.vs)
        self.pid_processor = PIDProcessor(knowledge_graph=self.kg)
        # Initialize empty chat history
        self.chat_history = []
        logger.info("PIDAssistant initialized with all components")

    def upload_pid(self, file_obj, use_gemini: bool = False) -> str:
        """Process uploaded P&ID file with optional Gemini model selection"""
        try:
            if file_obj is None:
                return "Please select a file to upload."
                
            file_path = file_obj.name if hasattr(file_obj, 'name') else None
            if not file_path:
                return "Invalid file object received"

            # Process the file using PIDProcessor with model selection
            results = self.pid_processor.process_file(file_path, use_gemini=use_gemini)
            
            if results['status'] == 'success':
                try:
                    self.vs.add_equipment(results['equipment'])
                    return f"Successfully processed with {'Gemini' if use_gemini else 'OpenAI'}.\nFound {len(results['equipment'])} equipment and {len(results['connections'])} connections."
                except Exception as e:
                    self.logger.error(f"Error adding to vector store: {e}")
                    return f"Error storing results: {str(e)}"
            else:
                return f"Error processing file: {results.get('message', 'Unknown error')}"
                
        except Exception as e:
            self.logger.error(f"Error in upload_pid: {e}", exc_info=True)
            return f"Error processing file: {str(e)}"

    def user_message(self, message: str, history: list) -> tuple:
        """Process user message and update chat history"""
        try:
            logger.info(f"Processing message: {message}")
            
            # Basic call without history for now
            response = self.nlp.process_query(message)
            logger.info(f"Processed query response: {response}")
            
            # Update history and clear input
            history.append((message, response))
            return "", history
            
        except Exception as e:
            logger.error(f"Error in user_message: {e}", exc_info=True)
            error_message = f"Error processing query: {str(e)}"
            history.append((message, error_message))
            return "", history

    def clear_chat(self):
        """Clear chat history and reset memory"""
        try:
            # Clear Gradio chat history
            self.chat_history = []
            # Reset LangChain conversation memory
            self.nlp.memory.clear()
            return None
            
        except Exception as e:
            logger.error(f"Error clearing chat: {e}")
            return None

# Create FastAPI app with Pydantic v2 config
app = FastAPI()

class ModelConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True
    )
    
def build_interface():
    """Create the Gradio interface"""
    assistant = PIDAssistant()
    
    with gr.Blocks(title="P&ID Knowledge Graph Assistant") as interface:
        with gr.Row():
            # Left column - File Upload
            with gr.Column(scale=1):
                gr.Markdown("## Upload P&ID")
                file_input = gr.File(
                    label="Upload P&ID File",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath"
                )
                model_selector = gr.Checkbox(
                    label="Use Gemini Pro Vision (if not checked, will use OpenAI)",
                    value=False,
                    interactive=True
                )
                status_output = gr.Textbox(
                    label="Processing Status",
                    interactive=False
                )
                upload_button = gr.Button("Process P&ID", variant="primary")
            
            # Right column - Chat Interface
            with gr.Column(scale=2):
                gr.Markdown("## Query P&ID")
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    show_label=True
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your query...",
                        container=False,
                        lines=1,
                        max_lines=1,
                        autofocus=True,
                        scale=4
                    )
                    clear = gr.Button("Clear", scale=1)
        
        # Event handlers with concurrency limits
        upload_button.click(
            fn=assistant.upload_pid,
            inputs=[file_input, model_selector],
            outputs=status_output,
            api_name="upload",
            concurrency_limit=1
        )
        
        msg_box.submit(
            fn=assistant.user_message,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot],
            api_name="chat",
            concurrency_limit=1
        )
        
        clear.click(
            fn=assistant.clear_chat,
            inputs=None,
            outputs=chatbot,
            api_name="clear",
            concurrency_limit=1
        )
    
    return interface

if __name__ == "__main__":
    logger.info("Starting P&ID Knowledge Graph Assistant")
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        debug=True,
        show_error=True,
        max_threads=4,
        auth=None,
        quiet=True
    )