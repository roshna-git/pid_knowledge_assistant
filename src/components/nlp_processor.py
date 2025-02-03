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
        """Process natural language query about P&ID."""
        try:
            # Analyze the query
            analysis = self._analyze_query(query)
            self.logger.info(f"Query analysis: {analysis}")
            
            if not analysis.get('equipment_ids'):
                return "No specific equipment mentioned in query."
            
            response_parts = []
            for eq_id in analysis['equipment_ids']:
                # Get equipment details
                equipment_info = self._get_equipment_info(eq_id)
                if equipment_info:
                    response_parts.append(f"Equipment {eq_id}:")
                    response_parts.append(f"Type: {equipment_info.get('type', 'Unknown')}")
                    
                    # Get connections
                    connected = self.knowledge_graph.get_connected_equipment(eq_id)
                    if connected:
                        response_parts.append("Connected to:")
                        for conn in connected:
                            response_parts.append(f"- {conn['id']} ({conn['type']})")
                    else:
                        response_parts.append("No connected equipment")
                else:
                    response_parts.append(f"No information found for {eq_id}")
                
                response_parts.append("")  # Add blank line between equipment
            
            return "\n".join(response_parts)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}"

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
        1. Query Type: EQUIPMENT_INFO, CONNECTIONS, PATH, SIMILARITY, or GENERAL
        2. Equipment IDs mentioned (e.g., T-101, P-101)
        3. Equipment Types mentioned (e.g., tank, pump, valve)
        4. Any specific relationships or conditions mentioned

        Format response as:
        Query Type: [type]
        Equipment IDs: [list of IDs]
        Equipment Types: [list of types]
        Relationships: [any mentioned relationships]"""

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
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            raise

    def _parse_analysis(self, response: str) -> Dict:
        """Parse OpenAI analysis response."""
        analysis = {
            'query_type': 'EQUIPMENT_INFO',
            'equipment_ids': [],
            'equipment_types': [],
            'relationships': []
        }
        
        try:
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Query Type:'):
                    analysis['query_type'] = line.split(':')[1].strip()
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
                            t.strip() for t in types.strip('[]').split(',')
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

    def _handle_equipment_info(self, analysis: Dict) -> Dict:
        """Handle queries about specific equipment."""
        result = {'equipment': [], 'related_info': []}
        
        # Check vector store for equipment information
        if analysis['equipment_ids']:
            for eq_id in analysis['equipment_ids']:
                eq_info = self.vector_store.get_equipment_by_id(eq_id)
                if eq_info:
                    result['equipment'].append(eq_info)
        
        # Check knowledge graph for additional information
        if self.knowledge_graph and analysis['equipment_ids']:
            for eq_id in analysis['equipment_ids']:
                connected = self.knowledge_graph.get_connected_equipment(eq_id)
                result['related_info'].extend(connected)
        
        return result

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

    def _handle_similarity_query(self, analysis: Dict) -> Dict:
        """Handle queries about similar equipment."""
        result = {'similar_equipment': []}
        
        if analysis['equipment_ids']:
            # Get equipment details
            eq_info = self.vector_store.get_equipment_by_id(analysis['equipment_ids'][0])
            if eq_info:
                # Find similar equipment
                similar = self.vector_store.query_similar(eq_info['document'])
                result['similar_equipment'] = similar
        
        return result

    def _handle_general_query(self, analysis: Dict) -> Dict:
        """Handle general queries about equipment."""
        result = {'equipment': [], 'summary': ''}
        
        # Search by equipment type if specified
        if analysis['equipment_types']:
            for eq_type in analysis['equipment_types']:
                equipment = self.vector_store.search_by_type(eq_type)
                result['equipment'].extend(equipment)
        
        return result