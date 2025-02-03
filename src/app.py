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
        logger.info("PIDAssistant initialized with all components")

    def upload_pid(self, file_obj) -> str:
        """Process uploaded P&ID file and store in both knowledge graph and vector store"""
        try:
            if file_obj is None:
                return "Please select a file to upload."
                
            # Handle file object from Gradio
            if hasattr(file_obj, 'name'):
                file_path = file_obj.name
            else:
                return "Invalid file object received"

            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing file: {file_path}")
            logger.info(f"File extension: {file_ext}")
            
            allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
            if file_ext not in allowed_extensions:
                return f"Invalid file type. Please upload one of these formats: {', '.join(allowed_extensions)}"

            try:
                # Process the file using PIDProcessor
                results = self.pid_processor.process_file(file_path)
                
                if results['status'] == 'success':
                    # Store equipment in vector database
                    try:
                        logger.info("Adding equipment to vector store...")
                        self.vs.add_equipment(results['equipment'])
                        logger.info("Equipment added to vector store successfully")
                    except Exception as e:
                        logger.error(f"Error adding to vector store: {e}")
                    
                    return (f"Successfully processed {os.path.basename(file_path)}.\n"
                           f"Found {len(results['equipment'])} equipment and "
                           f"{len(results['connections'])} connections.")
                else:
                    return f"Error processing file: {results.get('message', 'Unknown error')}"
                    
            except Exception as e:
                logger.error(f"Error in file processing: {e}", exc_info=True)
                return f"Error processing file: {str(e)}"
                    
        except Exception as e:
            logger.error(f"Error in upload_pid: {e}", exc_info=True)
            return f"Error processing file: {str(e)}"

    def user_message(self, message: str, history: list) -> tuple:
        """Process user message and update chat history"""
        try:
            logger.info(f"Processing message: {message}")
            
            # Process the query using NLP processor
            response = self.nlp.process_query(message)
            logger.info(f"Processed query: {response}")
            
            # Update history and clear input
            history.append((message, response))
            return "", history
            
        except Exception as e:
            logger.error(f"Error in user_message: {e}", exc_info=True)
            history.append((message, f"Error processing query: {str(e)}"))
            return "", history

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
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"]
                )
                status_output = gr.Textbox(
                    label="Processing Status",
                    interactive=False
                )
                upload_button = gr.Button("Process P&ID")
            
            # Right column - Chat Interface
            with gr.Column(scale=2):
                gr.Markdown("## Query P&ID")
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400
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
        
        # Event handlers
        msg_box.submit(
            fn=assistant.user_message,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot]
        )
        
        upload_button.click(
            fn=assistant.upload_pid,
            inputs=[file_input],
            outputs=[status_output]
        )
        
        clear.click(
            fn=lambda: None,
            inputs=None,
            outputs=chatbot
        )
    
    return interface

if __name__ == "__main__":
    logger.info("Starting P&ID Knowledge Graph Assistant")
    demo = build_interface()
    demo.launch(debug=True)