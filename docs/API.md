# API Documentation

## Components API Reference

### KnowledgeGraph

#### `KnowledgeGraph` Class
Primary interface for Neo4j database operations.

```python
class KnowledgeGraph:
    def __init__(self):
        """Initialize Neo4j connection."""

    def create_equipment(self, equipment_data: Dict[str, Any]) -> Dict:
        """Create equipment node in graph."""

    def create_connection(self, from_id: str, to_id: str, properties: Dict = None) -> Dict:
        """Create connection between equipment."""

    def get_equipment_by_type(self, equipment_type: str) -> List[Dict]:
        """Get all equipment of specific type."""

    def get_connected_equipment(self, equipment_id: str) -> List[Dict]:
        """Get equipment connected to specified equipment."""
```

### PIDProcessor

#### `PIDProcessor` Class
Handles P&ID image processing and analysis.

```python
class PIDProcessor:
    def __init__(self, knowledge_graph=None):
        """Initialize processor with optional knowledge graph."""

    def process_file(self, file_path: str) -> Dict:
        """Process P&ID file and extract equipment/connections."""

    def detect_equipment(self, image: np.ndarray) -> List[Dict]:
        """Detect equipment in processed image."""

    def detect_connections(self, image: np.ndarray, equipment_list: List[Dict]) -> List[Dict]:
        """Detect connections between equipment."""
```

### VectorStore

#### `VectorStore` Class
Manages vector embeddings and similarity search.

```python
class VectorStore:
    def __init__(self):
        """Initialize ChromaDB and embeddings."""

    def add_equipment(self, equipment_list: List[Dict]) -> None:
        """Add equipment to vector store."""

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find similar equipment using vector similarity."""

    def hybrid_search(self, query: str, equipment_type: Optional[str] = None) -> List[Dict]:
        """Combined vector and metadata search."""
```

### NLPProcessor

#### `NLPProcessor` Class
Handles natural language query processing.

```python
class NLPProcessor:
    def __init__(self, knowledge_graph=None, vector_store=None):
        """Initialize with knowledge graph and vector store."""

    def process_query(self, query: str) -> str:
        """Process natural language query."""

    def _analyze_query(self, query: str) -> Dict:
        """Analyze query intent and extract entities."""
```

## Response Formats

### Equipment Response
```json
{
    "id": "T-101",
    "type": "tank",
    "name": "Storage Tank",
    "description": "Main storage tank",
    "connections": [
        {
            "id": "P-101",
            "type": "pump",
            "relationship": "CONNECTS_TO"
        }
    ]
}
```

### Query Response
```json
{
    "status": "success",
    "equipment": [...],
    "connections": [...],
    "message": "Optional message"
}
```

## Error Handling

All components use consistent error handling:
- Specific exceptions for different error types
- Detailed error messages
- Proper logging of errors
- Graceful fallback behaviors

## Common Patterns

1. Equipment Creation:
```python
equipment_data = {
    'id': 'P-101',
    'type': 'pump',
    'name': 'Feed Pump',
    'description': 'Main feed pump'
}
result = knowledge_graph.create_equipment(equipment_data)
```

2. Query Processing:
```python
nlp = NLPProcessor(knowledge_graph, vector_store)
response = nlp.process_query("What equipment is connected to T-101?")
```

3. P&ID Processing:
```python
processor = PIDProcessor(knowledge_graph)
result = processor.process_file("path/to/pid.png")
```

## Best Practices

1. Always validate input data
2. Use proper error handling
3. Follow consistent naming conventions
4. Document code changes
5. Write tests for new functionality
