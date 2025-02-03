"""
Enhanced NLP Processor for P&ID queries
"""

import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

class NLPProcessor:
    def __init__(self, knowledge_graph=None, vector_store=None):
        self.logger = logging.getLogger(__name__)
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.openai_client = OpenAI()
        
        # Query patterns for common requests
        self.query_patterns = {
            'EQUIPMENT_INFO': r'tell me about|show|what is|describe',
            'CONNECTIONS': r'connected to|flows to|flows from|between',
            'EQUIPMENT_TYPE': r'all|show all|find all|list all',
            'PATH': r'path from|route from|flow from|between',
            'SIMILARITY': r'similar to|like|same as|equipment like'
        }

    def process_query(self, query: str) -> str:
        """Process natural language query about P&ID using both vector and graph search."""
        try:
            # Analyze the query
            analysis = self._analyze_query(query)
            self.logger.info(f"Query analysis: {analysis}")
            
            # Handle different query types
            if analysis['query_type'] == 'EQUIPMENT_INFO' and analysis['equipment_ids']:
                return self._handle_equipment_info(analysis)
            elif analysis['query_type'] == 'SIMILARITY' or analysis['keywords']:
                return self._handle_similarity_query(analysis, query)
            elif analysis['query_type'] == 'CONNECTIONS' and analysis['equipment_ids']:
                return self._handle_connections_query(analysis)
            else:
                # Handle semantic/general queries
                return self._handle_semantic_query(analysis, query)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def _handle_semantic_query(self, analysis: Dict, query: str) -> str:
        """Handle semantic queries using both vector search and equipment types."""
        response_parts = []
        
        # Use any mentioned equipment types
        if analysis['equipment_types']:
            response_parts.append(f"Finding equipment related to: {', '.join(analysis['equipment_types'])}")
            
            # Search graph for equipment types
            for eq_type in analysis['equipment_types']:
                equipment = self.knowledge_graph.get_equipment_by_type(eq_type)
                if equipment:
                    response_parts.append(f"\nFound {eq_type} equipment:")
                    for eq in equipment:
                        response_parts.append(f"- {eq['id']}: {eq.get('description', 'No description')}")
        
        # Use vector search for semantic similarity
        results = self.vector_store.hybrid_search(query)
        if results:
            if not response_parts:  # If no type-based results
                response_parts.append("Found related equipment:")
            else:
                response_parts.append("\nAdditional related equipment:")
            
            for item in results[:5]:  # Limit to top 5 results
                response_parts.append(f"- {item['id']} ({item['metadata']['type']})")
                if 'content' in item and item['content']:
                    # Add first line of content as preview
                    content_preview = item['content'].split('\n')[0]
                    response_parts.append(f"  {content_preview}")
        
        if not response_parts:
            return "No relevant equipment found."
            
        return "\n".join(response_parts)


    def _handle_connections_query(self, analysis: Dict) -> str:
        """Handle connection-related queries."""
        response_parts = []
        
        for eq_id in analysis.get('equipment_ids', []):
            connected = self.knowledge_graph.get_connected_equipment(eq_id)
            response_parts.append(f"Connections for {eq_id}:")
            
            if connected:
                for conn in connected:
                    response_parts.append(f"- {conn['id']} ({conn['type']})")
            else:
                response_parts.append("No connections found.")
                
            response_parts.append("")
            
        return "\n".join(response_parts)

    def _get_equipment_info(self, equipment_id: str) -> dict:
        """Get equipment information from knowledge graph."""
        try:
            # You might need to add this method to your KnowledgeGraph class
            query = """
            MATCH (e:Equipment {id: $id})
            RETURN e.id as id, e.type as type, e.name as name, e.description as description
            """
            
            with self.knowledge_graph.driver.session() as session:
                result = session.run(query, id=equipment_id)
                record = result.single()
                if record:
                    return {
                        'id': record['id'],
                        'type': record['type'],
                        'name': record['name'],
                        'description': record['description']
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting equipment info: {str(e)}")
            return None

    def _analyze_query(self, query: str) -> Dict:
        """Analyze query using OpenAI to determine intent and entities."""
        system_prompt = """You are a P&ID expert. Analyze the query and extract:
        1. Query Type: Choose one:
           - EQUIPMENT_INFO (for specific equipment queries)
           - SIMILARITY (for similarity or related equipment queries)
           - CONNECTIONS (for connection queries)
           - GENERAL (for broad or semantic queries)
        2. Equipment IDs mentioned (e.g., T-101, P-101)
        3. Equipment Types mentioned (e.g., tank, pump, valve)
        4. Keywords or concepts (e.g., pumping, flow, storage)
        5. Any specific relationships mentioned

        Format response as:
        Query Type: [type]
        Equipment IDs: [list of IDs]
        Equipment Types: [list of types]
        Keywords: [list of keywords]
        Relationships: [any mentioned relationships]
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=200
            )
            
            # Parse response
            analysis = self._parse_analysis(response.choices[0].message.content)
            self.logger.info(f"Query analysis: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            raise

    def _parse_analysis(self, response: str) -> Dict:
        """Parse OpenAI analysis response with enhanced parsing."""
        analysis = {
            'query_type': 'GENERAL',
            'equipment_ids': [],
            'equipment_types': [],
            'keywords': [],
            'relationships': []
        }
        
        try:
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('Query Type:'):
                    analysis['query_type'] = line.split(':')[1].strip().upper()
                elif line.startswith('Equipment IDs:'):
                    ids = line.split(':')[1].strip()
                    if ids and ids != '[]':
                        analysis['equipment_ids'] = [
                            id.strip() for id in ids.strip('[]').split(',')
                        ]
                elif line.startswith('Equipment Types:'):
                    types = line.split(':')[1].strip()
                    if types and types != '[]':
                        analysis['equipment_types'] = [
                            t.strip().lower() for t in types.strip('[]').split(',')
                        ]
                elif line.startswith('Keywords:'):
                    keywords = line.split(':')[1].strip()
                    if keywords and keywords != '[]':
                        analysis['keywords'] = [
                            k.strip().lower() for k in keywords.strip('[]').split(',')
                        ]
                elif line.startswith('Relationships:'):
                    rels = line.split(':')[1].strip()
                    if rels and rels != '[]':
                        analysis['relationships'] = [
                            r.strip() for r in rels.strip('[]').split(',')
                        ]
            
            self.logger.info(f"Parsed analysis: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error parsing analysis: {str(e)}")
            return analysis

    def _handle_equipment_info(self, analysis: Dict) -> str:
        """Handle specific equipment information queries."""
        response_parts = []
        
        if analysis.get('equipment_ids'):
            for eq_id in analysis['equipment_ids']:
                # Get equipment details from graph
                equipment_info = self.knowledge_graph.get_equipment_by_id(eq_id)
                if equipment_info:
                    response_parts.append(f"Equipment {eq_id}:")
                    response_parts.append(f"Type: {equipment_info.get('type', 'Unknown')}")
                    
                    # Get connections
                    connected = self.knowledge_graph.get_connected_equipment(eq_id)
                    if connected:
                        response_parts.append("Connected to:")
                        for conn in connected:
                            response_parts.append(f"- {conn['id']} ({conn['type']})")
                            
                    # Get similar equipment
                    if self.vector_store:
                        similar = self.vector_store.get_similar_equipment(eq_id, n_results=3)
                        if similar:
                            response_parts.append("\nSimilar equipment:")
                            for sim in similar:
                                response_parts.append(f"- {sim['id']} ({sim['metadata']['type']})")
                else:
                    response_parts.append(f"No information found for {eq_id}")
                
                response_parts.append("")  # Add blank line between equipment
        else:
            response_parts.append("No specific equipment mentioned in query.")
            
        return "\n".join(response_parts)

    def _handle_connections(self, analysis: Dict) -> Dict:
        """Handle queries about equipment connections."""
        result = {'connections': [], 'equipment': []}
        
        if self.knowledge_graph and analysis['equipment_ids']:
            # Get direct connections for specified equipment
            for eq_id in analysis['equipment_ids']:
                connected = self.knowledge_graph.get_connected_equipment(eq_id)
                result['connections'].extend(connected)
                
                # Get equipment details from vector store
                eq_info = self.vector_store.get_equipment_by_id(eq_id)
                if eq_info:
                    result['equipment'].append(eq_info)
        
        return result

    def _handle_path_query(self, analysis: Dict) -> Dict:
        """Handle queries about paths between equipment."""
        result = {'path': [], 'equipment': []}
        
        if self.knowledge_graph and len(analysis['equipment_ids']) >= 2:
            start_id = analysis['equipment_ids'][0]
            end_id = analysis['equipment_ids'][1]
            
            # Get path from knowledge graph
            path = self.knowledge_graph.get_flow_path(start_id, end_id)
            result['path'] = path
            
            # Get equipment details for path
            for eq_id in path.get('path_ids', []):
                eq_info = self.vector_store.get_equipment_by_id(eq_id)
                if eq_info:
                    result['equipment'].append(eq_info)
        
        return result

    def _handle_similarity_query(self, analysis: Dict, query: str) -> str:
        """Handle similarity-based queries."""
        response_parts = []
        
        if analysis.get('equipment_ids'):
            # Find similar equipment to specified ID
            eq_id = analysis['equipment_ids'][0]
            similar = self.vector_store.get_similar_equipment(eq_id)
            
            response_parts.append(f"Equipment similar to {eq_id}:")
            for item in similar:
                response_parts.append(f"- {item['id']} ({item['metadata']['type']})")
                
        else:
            # Use semantic search based on query
            equipment_type = analysis.get('equipment_types', [None])[0]
            results = self.vector_store.hybrid_search(query, equipment_type)
            
            response_parts.append("Found relevant equipment:")
            for item in results:
                response_parts.append(f"- {item['id']} ({item['metadata']['type']})")
                
        return "\n".join(response_parts)

    def _handle_general_query(self, query: str) -> str:
        """Handle general queries using vector search."""
        results = self.vector_store.hybrid_search(query)
        
        if not results:
            return "No relevant equipment found."
            
        response_parts = ["Relevant equipment found:"]
        for item in results:
            response_parts.append(f"- {item['id']} ({item['metadata']['type']})")
            content_preview = item['content'].split('\n')[0]  # First line only
            response_parts.append(f"  {content_preview}")
            
        return "\n".join(response_parts)