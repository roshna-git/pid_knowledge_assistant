# P&ID Knowledge Graph Assistant

An intelligent system for processing and querying Piping and Instrumentation Diagrams (P&IDs) using computer vision, graph databases, and natural language processing.

## Features

- **P&ID Processing**: Automatic detection of equipment and connections from P&ID diagrams
- **Knowledge Graph**: Neo4j-based storage of P&ID elements and relationships
- **Vector Search**: Semantic search capabilities using ChromaDB
- **Natural Language Interface**: Query P&IDs using natural language
- **Web Interface**: Easy-to-use Gradio-based interface

## Prerequisites

- Python 3.8+
- Neo4j Database
- OpenAI API Key
- ChromaDB

## Installation

1. Clone the repository:
```bash
git clone https://github.com/roshna-git/pid_knowledge_assistant.git
cd pid_knowledge_assistant
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: new_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env` file:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the application:
```bash
python src/app.py
```

2. Access the web interface at `http://localhost:7860`

3. Upload a P&ID file:
   - Supported formats: PDF, PNG, JPG, JPEG
   - Click "Process P&ID" to analyze the diagram

4. Query the P&ID:
   - Ask about specific equipment (e.g., "What is T-101?")
   - Find connected equipment (e.g., "What's connected to P-101?")
   - Search by type (e.g., "Show me all pumps")
   - Find similar equipment (e.g., "Show equipment related to pumping")

## Project Structure

```
pid_knowledge_assistant/
├── src/
│   ├── components/
│   │   ├── knowledge_graph.py   # Neo4j interface
│   │   ├── pid_processor.py     # P&ID processing
│   │   ├── nlp_processor.py     # Query processing
│   │   └── vector_store.py      # ChromaDB interface
│   ├── data/                    # Data storage
│   ├── static/                  # Static files
│   └── app.py                   # Main application
├── tests/                       # Test files
├── docs/                        # Documentation
└── requirements.txt            # Dependencies
```

## API Documentation

See [API.md](docs/API.md) for detailed API documentation.

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for development guidelines.

