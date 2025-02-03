"""
Vector Store Component using ChromaDB for equipment similarity search
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            # Create persistent ChromaDB client
            db_path = Path("chroma_db")
            db_path.mkdir(exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Initialize OpenAI embeddings
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-3-small"
            )
            
            # Create or get collection with embeddings
            self.collection = self.client.get_or_create_collection(
                name="pid_equipment",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            self.logger.info("Vector store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _create_document_text(self, equipment: Dict) -> str:
        """Create rich text representation for equipment."""
        doc_parts = [
            f"Equipment ID: {equipment['id']}",
            f"Type: {equipment['type']}",
            f"Description: {equipment.get('description', 'No description available')}"
        ]
        
        # Add specifications if available
        if 'specifications' in equipment:
            doc_parts.append("Specifications:")
            for key, value in equipment['specifications'].items():
                doc_parts.append(f"- {key}: {value}")
        
        # Add connections if available
        if 'connections' in equipment:
            doc_parts.append("Connected to:")
            for conn in equipment['connections']:
                doc_parts.append(f"- {conn['id']} ({conn['type']})")
        
        return "\n".join(doc_parts)

    def add_equipment(self, equipment_list: List[Dict]) -> None:
        """Add equipment to vector store with enhanced embeddings."""
        try:
            documents = []
            ids = []
            metadatas = []
            
            for equipment in equipment_list:
                # Create rich text representation
                doc_text = self._create_document_text(equipment)
                
                # Store document and metadata
                documents.append(doc_text)
                ids.append(equipment['id'])
                metadatas.append({
                    'type': equipment['type'],
                    'id': equipment['id']
                })
            
            # Add to collection
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(equipment_list)} equipment to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding equipment to vector store: {str(e)}")
            raise

    def query_similar(self, query: str, n_results: int = 5, 
                     filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """Query for similar equipment using vector similarity."""
        try:
            # Create query with optional filters
            query_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if filter_criteria:
                query_params["where"] = filter_criteria
            
            # Execute query
            results = self.collection.query(**query_params)
            
            # Format results
            similar_equipment = []
            if results['ids']:
                for idx, eq_id in enumerate(results['ids'][0]):
                    similar_equipment.append({
                        'id': eq_id,
                        'metadata': results['metadatas'][0][idx],
                        'content': results['documents'][0][idx],
                        'similarity': float(results['distances'][0][idx]) if 'distances' in results else None
                    })
            
            return similar_equipment
            
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise

    def hybrid_search(self, query: str, equipment_type: Optional[str] = None) -> List[Dict]:
        """Perform hybrid search using both vector similarity and metadata filtering."""
        try:
            filter_criteria = {"type": equipment_type} if equipment_type else None
            
            # Get vector similarity results
            vector_results = self.query_similar(
                query=query,
                n_results=10,
                filter_criteria=filter_criteria
            )
            
            return vector_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def update_equipment(self, equipment_id: str, new_data: Dict) -> None:
        """Update equipment in vector store."""
        try:
            # Create new document text
            doc_text = self._create_document_text(new_data)
            
            # Update in collection
            self.collection.update(
                ids=[equipment_id],
                documents=[doc_text],
                metadatas=[{
                    'type': new_data['type'],
                    'id': new_data['id']
                }]
            )
            
            self.logger.info(f"Updated equipment {equipment_id} in vector store")
            
        except Exception as e:
            self.logger.error(f"Error updating equipment: {str(e)}")
            raise

    def delete_equipment(self, equipment_id: str) -> None:
        """Delete equipment from vector store."""
        try:
            self.collection.delete(ids=[equipment_id])
            self.logger.info(f"Deleted equipment {equipment_id} from vector store")
            
        except Exception as e:
            self.logger.error(f"Error deleting equipment: {str(e)}")
            raise

    def get_similar_equipment(self, equipment_id: str, n_results: int = 5) -> List[Dict]:
        """Find equipment similar to a specific equipment ID."""
        try:
            # Get equipment document
            result = self.collection.get(ids=[equipment_id])
            
            if not result['documents']:
                return []
            
            # Use document as query
            return self.query_similar(
                query=result['documents'][0],
                n_results=n_results + 1  # Add 1 to account for self-match
            )[1:]  # Remove self-match from results
            
        except Exception as e:
            self.logger.error(f"Error finding similar equipment: {str(e)}")
            raise