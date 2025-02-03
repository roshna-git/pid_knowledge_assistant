import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# LLM Configuration
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-2')
LLM_API_KEY = os.getenv('LLM_API_KEY')

# ChromaDB Configuration
CHROMADB_PATH = os.getenv('CHROMADB_PATH', './data/chromadb')