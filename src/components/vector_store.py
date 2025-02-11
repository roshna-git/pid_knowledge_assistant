"""
Vector Store Component using LangChain and ChromaDB for equipment similarity search
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class VectorStore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            # Create persistent directory for ChromaDB
            db_path = Path("chroma_db")
            db_path.mkdir(exist_ok=True)
            
            # Initialize LangChain embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Initialize text splitter for document chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
            
            # Create or get Chroma collection using LangChain
            self.vectorstore = Chroma(
                persist_directory=str(db_path),
                embedding_function=self.embeddings,
                collection_name="pid_equipment"
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
        """Add equipment to vector store with LangChain documents."""
        try:
            documents = []
            
            for equipment in equipment_list:
                # Create rich text representation
                doc_text = self._create_document_text(equipment)
                
                # Create LangChain document
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        'id': equipment['id'],
                        'type': equipment['type'],
                        'name': equipment.get('name', ''),
                        'description': equipment.get('description', '')
                    }
                )
                documents.append(doc)
            
            # Split documents into chunks if needed
            chunked_docs = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vectorstore.add_documents(documents=chunked_docs)
            
            self.logger.info(f"Added {len(equipment_list)} equipment to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding equipment to vector store: {str(e)}")
            raise

    def query_similar(self, query: str, n_results: int = 5, 
                     filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """Query for similar equipment using LangChain retriever."""
        try:
            # Prepare search filters
            search_kwargs = {"k": n_results}
            if filter_criteria:
                search_kwargs["filter"] = filter_criteria
            
            # Get relevant documents
            docs = self.vectorstore.similarity_search(
                query,
                **search_kwargs
            )
            
            # Format results
            similar_equipment = []
            for doc in docs:
                similar_equipment.append({
                    'id': doc.metadata.get('id', ''),
                    'type': doc.metadata.get('type', ''),
                    'name': doc.metadata.get('name', ''),
                    'description': doc.metadata.get('description', ''),
                    'content': doc.page_content
                })
            
            return similar_equipment
            
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise

    def hybrid_search(self, query: str, equipment_type: Optional[str] = None) -> List[Dict]:
        """Perform hybrid search using LangChain retriever."""
        try:
            filter_criteria = {"type": equipment_type} if equipment_type else None
            return self.query_similar(
                query=query,
                n_results=10,
                filter_criteria=filter_criteria
            )
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def get_similar_equipment(self, equipment_id: str, n_results: int = 5) -> List[Dict]:
        """Find equipment similar to a specific equipment ID."""
        try:
            # Search for equipment by ID
            docs = self.vectorstore.similarity_search(
                f"Equipment ID: {equipment_id}",
                k=1
            )
            
            if not docs:
                return []
            
            # Use document content to find similar equipment
            similar_docs = self.vectorstore.similarity_search(
                docs[0].page_content,
                k=n_results + 1  # Add 1 to account for self-match
            )
            
            # Format and filter results
            similar_equipment = []
            for doc in similar_docs:
                if doc.metadata.get('id') != equipment_id:  # Exclude self-match
                    similar_equipment.append({
                        'id': doc.metadata.get('id', ''),
                        'type': doc.metadata.get('type', ''),
                        'content': doc.page_content
                    })
            
            return similar_equipment[:n_results]
            
        except Exception as e:
            self.logger.error(f"Error finding similar equipment: {str(e)}")
            raise

    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get LangChain retriever interface."""
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs if search_kwargs else {"k": 5}
        )