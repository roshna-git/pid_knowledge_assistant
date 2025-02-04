# test_setup.py
import gradio as gr
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
import chromadb
import os
from dotenv import load_dotenv

def test_components():
    results = []
    
    # Load environment variables
    load_dotenv()
    
    # Test 1: Gradio
    results.append("✓ Gradio is working")
    
    # Test 2: Environment Variables
    api_key = os.getenv('OPENAI_API_KEY')
    results.append(f"OpenAI API Key {'is set' if api_key else 'is not set'}")
    
    # Test 3: OpenAI API
    try:
        llm = OpenAI(api_key=api_key)
        results.append("✓ LangChain OpenAI integration is set up")
    except Exception as e:
        results.append(f"✗ OpenAI setup error: {str(e)}")
    
    # Test 4: ChromaDB
    try:
        chroma_client = chromadb.Client()
        results.append("✓ ChromaDB is working")
    except Exception as e:
        results.append(f"✗ ChromaDB error: {str(e)}")
    
    return "\n".join(results)

demo = gr.Interface(
    fn=test_components, 
    inputs=None, 
    outputs="text",
    title="Component Test"
)

if __name__ == "__main__":
    demo.launch()