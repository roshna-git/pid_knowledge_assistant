# rag_test.py
import gradio as gr
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_vector_store():
    logger.info("Initializing vector store...")
    try:
        pid_data = [
            "Equipment T-101 is a storage tank for raw materials with a capacity of 1000L.",
            "Pump P-101 is a centrifugal pump that transfers material from T-101.",
            "Valve V-101 is a control valve regulating flow between P-101 and T-102.",
            "Tank T-102 is a product storage tank connected to V-101."
        ]
        
        logger.info("Creating embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        logger.info("Splitting text...")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(pid_data)
        
        logger.info("Creating Chroma database...")
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name="pid_data",
            persist_directory="./data/chroma"
        )
        
        logger.info("Vector store initialized successfully")
        return vectordb
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

def process_query(query):
    logger.info(f"Processing query: {query}")
    try:
        # Verify API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Error: OpenAI API key not found"
        
        logger.info("Initializing LLM...")
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0,
            max_tokens=500
        )
        
        logger.info("Getting vector store...")
        vectordb = initialize_vector_store()
        
        logger.info("Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )
        
        logger.info("Getting response from chain...")
        response = qa_chain({"query": query})
        
        answer = response['result']
        sources = [doc.page_content for doc in response['source_documents']]
        
        logger.info("Successfully processed query")
        return f"Answer: {answer}\n\nSources:\n" + "\n".join(f"- {src}" for src in sources)
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Create Gradio interface with error handling
def safe_process_query(query):
    try:
        return process_query(query)
    except Exception as e:
        return f"An error occurred: {str(e)}"

demo = gr.Interface(
    fn=safe_process_query,
    inputs=gr.Textbox(
        label="Enter your question",
        placeholder="e.g., What is T-101?",
        lines=2
    ),
    outputs=gr.Textbox(label="Response", lines=10),
    title="P&ID Query System",
    description="Ask questions about P&ID equipment and connections"
)

if __name__ == "__main__":
    os.makedirs("./data/chroma", exist_ok=True)
    print("Starting Gradio interface...")
    print(f"OpenAI API Key present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    demo.launch(share=False)