"""
Enhanced NLP Processor for P&ID queries with LangChain RAG integration
"""

import logging
from typing import Dict, List, Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from .prompt_templates import get_prompt_templates

class NLPProcessor:
    def __init__(self, knowledge_graph=None, vector_store=None):
        self.logger = logging.getLogger(__name__)
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Initialize LangChain components with enhanced settings
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            max_tokens=1000
        )
        
        # Enhanced conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Load enhanced prompts
        self.prompts = get_prompt_templates()
        
        # Initialize RAG components
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize enhanced RAG components."""
        try:
            if not self.vector_store:
                self.logger.warning("No vector store available, RAG functionality will be limited")
                self.chain = None
                return

            # Create main QA chain with simpler configuration
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            self.logger.info("Enhanced RAG chains initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG components: {str(e)}")
            self.chain = None

    def process_query(self, query: str, chat_history: Optional[List] = None) -> str:
        """Enhanced query processing with specialized handling."""
        try:
            # Analyze query intent
            analysis = self._analyze_query(query)
            self.logger.info(f"Query analysis: {analysis}")
            
            # Get context and graph data
            vector_context = self._get_vector_context(query, analysis)
            graph_data = self._get_graph_data(analysis)
            
            # Handle different query types
            if analysis['query_type'] == 'EQUIPMENT_INFO':
                return self._process_equipment_query(query, vector_context, graph_data, analysis)
            elif analysis['query_type'] == 'CONNECTIONS':
                return self._process_connection_query(analysis)
            elif analysis['query_type'] == 'PATH':
                return self._process_flow_query(query, vector_context, graph_data, analysis)
            else:
                # Use main RAG chain for general queries
                return self._process_general_query(query, vector_context, graph_data, chat_history)

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self._format_error_response(str(e))

    def _analyze_query(self, query: str) -> Dict:
        """Analyze query using OpenAI to determine intent and entities."""
        system_prompt = """You are a P&ID expert. Analyze the query and extract:
        1. Query Type: Choose one:
           - EQUIPMENT_INFO (for specific equipment queries)
           - SIMILARITY (for similarity or related equipment queries)
           - CONNECTIONS (for connection queries)
           - PATH (for flow path queries)
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
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200
            )
            
            return self._parse_analysis(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            return {
                'query_type': 'GENERAL',
                'equipment_ids': [],
                'equipment_types': [],
                'keywords': [],
                'relationships': []
            }

    def _parse_analysis(self, response: str) -> Dict:
        """Parse OpenAI analysis response."""
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
                        analysis['equipment_ids'] = [id.strip() for id in ids.strip('[]').split(',')]
                elif line.startswith('Equipment Types:'):
                    types = line.split(':')[1].strip()
                    if types and types != '[]':
                        analysis['equipment_types'] = [t.strip().lower() for t in types.strip('[]').split(',')]
                elif line.startswith('Keywords:'):
                    keywords = line.split(':')[1].strip()
                    if keywords and keywords != '[]':
                        analysis['keywords'] = [k.strip().lower() for k in keywords.strip('[]').split(',')]
                elif line.startswith('Relationships:'):
                    rels = line.split(':')[1].strip()
                    if rels and rels != '[]':
                        analysis['relationships'] = [r.strip() for r in rels.strip('[]').split(',')]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error parsing analysis: {str(e)}")
            return analysis

    def _get_vector_context(self, query: str, analysis: Dict) -> str:
        """Get enhanced vector store context."""
        try:
            if not self.vector_store:
                return ""

            # Determine search parameters based on query type
            search_kwargs = {"k": 5}
            if analysis['equipment_ids']:
                search_kwargs["filter"] = {"id": {"$in": analysis['equipment_ids']}}
            elif analysis['equipment_types']:
                search_kwargs["filter"] = {"type": {"$in": analysis['equipment_types']}}

            results = self.vector_store.query_similar(
                query=query,
                n_results=5,
                filter_criteria=search_kwargs.get("filter")
            )
            
            return "\n\n".join(r.get('content', '') for r in results if 'content' in r)
            
        except Exception as e:
            self.logger.error(f"Error getting vector context: {str(e)}")
            return ""

    def _get_graph_data(self, analysis: Dict) -> str:
        """Safely get relevant data from knowledge graph"""
        try:
            data_parts = []
            
            # Get equipment information
            if analysis.get('equipment_ids'):
                for eq_id in analysis['equipment_ids']:
                    eq_info = self.knowledge_graph.get_equipment_by_id(eq_id)
                    if eq_info:
                        data_parts.append(f"Equipment {eq_id}: {eq_info}")

            # Get equipment by type
            if analysis.get('equipment_types'):
                for eq_type in analysis['equipment_types']:
                    equipment = self.knowledge_graph.get_equipment_by_type(eq_type)
                    if equipment:
                        data_parts.append(f"{eq_type.capitalize()} equipment: {equipment}")

            return "\n".join(data_parts) if data_parts else "No specific graph data available."
            
        except Exception as e:
            self.logger.error(f"Error getting graph data: {str(e)}")
            return "Error retrieving graph data."

    def _process_equipment_query(self, query: str, context: str, graph_data: str, analysis: Dict) -> str:
        """Process equipment-specific queries."""
        try:
            if not analysis.get('equipment_ids'):
                # Handle general equipment listing
                all_equipment = []
                for eq_type in ['pump', 'tank', 'valve']:
                    equipment = self.knowledge_graph.get_equipment_by_type(eq_type)
                    all_equipment.extend(equipment)
                
                if all_equipment:
                    response_parts = ["Here is the equipment in the system:"]
                    for eq in all_equipment:
                        response_parts.append(f"- {eq['id']} ({eq['type']})")
                    return "\n".join(response_parts)
            
            # Use equipment chain for specific equipment
            response = self.equipment_chain.predict(
                context=context,
                graph_data=graph_data,
                question=query
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing equipment query: {str(e)}")
            return self._handle_equipment_info(analysis)  # Fallback to traditional processing

    def _process_connection_query(self, analysis: Dict) -> str:
        """Process connection-specific queries with enhanced details."""
        try:
            response_parts = []
            
            if not analysis.get('equipment_ids'):
                # Get all connections
                all_equipment = []
                for eq_type in ['pump', 'tank', 'valve']:
                    equipment = self.knowledge_graph.get_equipment_by_type(eq_type)
                    all_equipment.extend(equipment)
                
                for eq in all_equipment:
                    connections = self.knowledge_graph.get_connected_equipment(eq['id'])
                    if connections:
                        response_parts.append(f"{eq['id']} ({eq['type']}) is connected to:")
                        for conn in connections:
                            response_parts.append(f"  - {conn['id']} ({conn['type']})")
                        response_parts.append("")  # Add spacing
            else:
                # Get specific connections
                for eq_id in analysis['equipment_ids']:
                    connections = self.knowledge_graph.get_connected_equipment(eq_id)
                    if connections:
                        response_parts.append(f"Connections for {eq_id}:")
                        for conn in connections:
                            response_parts.append(f"- Connected to {conn['id']} ({conn['type']})")
                            if 'properties' in conn:
                                for prop, value in conn['properties'].items():
                                    response_parts.append(f"  - {prop}: {value}")
                        response_parts.append("")  # Add spacing
            
            return "\n".join(response_parts) if response_parts else "No connection information found."
            
        except Exception as e:
            self.logger.error(f"Error processing connection query: {str(e)}")
            return self._handle_connections_query(analysis)  # Fallback

    def _process_flow_query(self, query: str, context: str, graph_data: str, analysis: Dict) -> str:
        """Process flow path queries with enhanced details."""
        try:
            # Use flow chain for processing
            response = self.flow_chain.predict(
                context=context,
                graph_data=graph_data,
                question=query
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing flow query: {str(e)}")
            return self._handle_flow_path_query(analysis)  # Fallback

    def _process_general_query(self, query: str, context: str, graph_data: str, 
                             chat_history: Optional[List] = None) -> str:
        """Process general queries using main RAG chain."""
        try:
            if self.chain:
                # Simplified chain call with just question and chat_history
                response = self.chain({
                    "question": query,
                    "chat_history": chat_history or []
                })
                return response['answer']
            else:
                # Fallback to traditional processing
                return self._traditional_process_query(query, self._analyze_query(query))
                
        except Exception as e:
            self.logger.error(f"Error in general query processing: {str(e)}")
            return self._handle_equipment_info(self._analyze_query(query))  # Fallback to basic equipment info

    def _handle_equipment_info(self, analysis: Dict) -> str:
        """Traditional handler for equipment information queries."""
        response_parts = []
        
        if analysis.get('equipment_ids'):
            for eq_id in analysis['equipment_ids']:
                equipment_info = self.knowledge_graph.get_equipment_by_id(eq_id)
                if equipment_info:
                    response_parts.append(f"Equipment {eq_id}:")
                    response_parts.append(f"Type: {equipment_info.get('type', 'Unknown')}")
                    response_parts.append(f"Description: {equipment_info.get('description', 'No description available')}")
                    
                    # Get connections
                    connected = self.knowledge_graph.get_connected_equipment(eq_id)
                    if connected:
                        response_parts.append("Connected to:")
                        for conn in connected:
                            response_parts.append(f"- {conn['id']} ({conn['type']})")
                else:
                    response_parts.append(f"No information found for {eq_id}")
                response_parts.append("")  # Add blank line between equipment
                
        return "\n".join(response_parts) if response_parts else "No specific equipment information found."

    def _handle_connections_query(self, analysis: Dict) -> str:
        """Traditional handler for connection queries."""
        try:
            response_parts = []
            
            for eq_id in analysis.get('equipment_ids', []):
                connected = self.knowledge_graph.get_connected_equipment(eq_id)
                if connected:
                    response_parts.append(f"Connections for {eq_id}:")
                    for conn in connected:
                        response_parts.append(f"- {conn['id']} ({conn['type']})")
                else:
                    response_parts.append(f"No connections found for {eq_id}")
                response_parts.append("")
                
            return "\n".join(response_parts) if response_parts else "No connection information found."
            
        except Exception as e:
            self.logger.error(f"Error handling connections query: {str(e)}")
            return "I encountered an error while retrieving connection information."

    def _handle_flow_path_query(self, analysis: Dict) -> str:
        """Traditional handler for flow path queries."""
        try:
            if len(analysis.get('equipment_ids', [])) < 2:
                return "Please specify both the start and end equipment to find the flow path."

            start_id = analysis['equipment_ids'][0]
            end_id = analysis['equipment_ids'][1]
            
            paths = self.knowledge_graph.get_flow_path(start_id, end_id)
            
            if paths and paths[0].get('path_ids'):
                path_str = " -> ".join(paths[0]['path_ids'])
                return f"The flow path from {start_id} to {end_id} is: {path_str}"
            else:
                return f"No flow path found between {start_id} and {end_id}"
                
        except Exception as e:
            self.logger.error(f"Error handling flow path query: {str(e)}")
            return "I encountered an error while finding the flow path."

    def _traditional_process_query(self, query: str, analysis: Dict) -> str:
        """Traditional query processing as fallback."""
        try:
            if analysis['query_type'] == 'EQUIPMENT_INFO':
                if analysis['equipment_ids']:
                    return self._handle_equipment_info(analysis)
                else:
                    # List all equipment
                    all_equipment = []
                    for eq_type in ['pump', 'tank', 'valve']:
                        equipment = self.knowledge_graph.get_equipment_by_type(eq_type)
                        all_equipment.extend(equipment)
                    
                    if all_equipment:
                        response_parts = ["Here is the equipment in the system:"]
                        for eq in all_equipment:
                            response_parts.append(f"- {eq['id']} ({eq['type']})")
                        return "\n".join(response_parts)
                    return "No equipment found in the system."
            elif analysis['query_type'] == 'CONNECTIONS':
                return self._handle_connections_query(analysis)
            elif analysis['query_type'] == 'PATH':
                return self._handle_flow_path_query(analysis)
            else:
                return "I'm not sure how to process that query. Could you be more specific about what equipment information you're looking for?"
                
        except Exception as e:
            self.logger.error(f"Error in traditional processing: {str(e)}")
            return "I encountered an error processing your query. Please try asking about specific equipment or connections."

    def _format_error_response(self, error: str) -> str:
        """Format user-friendly error messages."""
        if 'metadata' in error.lower():
            return "I'm having trouble retrieving some information. Could you please rephrase your question or ask about specific equipment?"
        return "I encountered an error processing your query. Please try again with a more specific question."