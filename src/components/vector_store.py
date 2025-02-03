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
        
        # Initialize ChromaDB
        try:
            # Create a persistent client with a local database
            db_path = Path("chroma_db")
            db_path.mkdir(exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Use OpenAI embeddings
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-3-small"
            )
            
            # Create or get collection for P&ID equipment
            self.collection = self.client.get_or_create_collection(
                name="pid_equipment",
                embedding_function=self.embedding_function
            )
            
            self.logger.info("Vector store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def add_equipment(self, equipment_list: List[Dict]):
        """Add equipment to vector store."""
        try:
            for equipment in equipment_list:
                # Create document text that captures equipment properties
                doc_text = (
                    f"Equipment Type: {equipment['type']} "
                    f"ID: {equipment['id']} "
                    f"Description: {equipment.get('description', '')} "
                )
                
                # Add to ChromaDB
                self.collection.add(
                    documents=[doc_text],
                    ids=[equipment['id']],
                    metadatas=[{
                        'type': equipment['type'],
                        'id': equipment['id']
                    }]
                )
                
            self.logger.info(f"Added {len(equipment_list)} equipment to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding equipment to vector store: {str(e)}")
            raise

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query for similar equipment using vector similarity."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Extract and format results
            similar_equipment = []
            if results['ids']:
                for idx, eq_id in enumerate(results['ids'][0]):
                    similar_equipment.append({
                        'id': eq_id,
                        'metadata': results['metadatas'][0][idx],
                        'document': results['documents'][0][idx],
                        'distance': results.get('distances', [[]])[0][idx] if results.get('distances') else None
                    })
            
            return similar_equipment
            
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise

    def search_by_type(self, equipment_type: str) -> List[Dict]:
        """Search equipment by type."""
        try:
            results = self.collection.query(
                query_texts=[f"Equipment Type: {equipment_type}"],
                where={"type": equipment_type},
                n_results=10
            )
            
            equipment_list = []
            if results['ids']:
                for idx, eq_id in enumerate(results['ids'][0]):
                    equipment_list.append({
                        'id': eq_id,
                        'metadata': results['metadatas'][0][idx],
                        'document': results['documents'][0][idx]
                    })
            
            return equipment_list
            
        except Exception as e:
            self.logger.error(f"Error searching by type: {str(e)}")
            raise

    def get_equipment_by_id(self, equipment_id: str) -> Optional[Dict]:
        """Get equipment by ID."""
        try:
            results = self.collection.get(
                ids=[equipment_id]
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'metadata': results['metadatas'][0],
                    'document': results['documents'][0]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting equipment by ID: {str(e)}")
            raise

    def delete_equipment(self, equipment_id: str):
        """Delete equipment from vector store."""
        try:
            self.collection.delete(
                ids=[equipment_id]
            )
            self.logger.info(f"Deleted equipment {equipment_id} from vector store")
            
        except Exception as e:
            self.logger.error(f"Error deleting equipment: {str(e)}")
            raise

    def clear_all(self):
        """Clear all equipment from vector store."""
        try:
            self.client.delete_collection("pid_equipment")
            self.collection = self.client.create_collection(
                name="pid_equipment",
                embedding_function=self.embedding_function
            )
            self.logger.info("Cleared all equipment from vector store")
            
        except Exception as e:
            self.logger.error(f"Error clearing vector store: {str(e)}")
            raise