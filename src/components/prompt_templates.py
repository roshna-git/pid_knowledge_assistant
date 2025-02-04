"""
Prompt templates for P&ID knowledge assistant
"""

from langchain.prompts import PromptTemplate

# Main QA prompt with structured context handling
MAIN_QA_TEMPLATE = """You are a P&ID (Piping and Instrumentation Diagram) knowledge assistant. Use the provided context to answer questions about equipment, connections, and processes.

Context from Knowledge Base:
{context}

Graph Database Information:
{graph_data}

Previous Conversation:
{chat_history}

Given Question: {question}

Instructions:
1. Focus on technical accuracy and clarity
2. If equipment is mentioned, include its specifications and connections
3. If discussing flow or processes, describe the sequence clearly
4. If information is missing, specify what details are not available
5. Keep responses concise but complete

Response: Let me help you understand the P&ID system."""

# Equipment-specific prompt for detailed information
EQUIPMENT_TEMPLATE = """Analyze the following equipment information from a P&ID system:

Equipment Details:
{context}

Connection Information:
{graph_data}

Question about the equipment: {question}

Instructions:
1. Describe the equipment's main characteristics
2. List all connections and relationships
3. Include relevant technical specifications
4. Mention any safety or operational considerations
5. State if any critical information is missing

Response: Here's what I know about this equipment."""

# Process flow prompt for path analysis
FLOW_TEMPLATE = """Analyze the flow path in the P&ID system:

System Context:
{context}

Connection Data:
{graph_data}

Flow Path Question: {question}

Instructions:
1. Trace the flow path step by step
2. Identify all equipment in the path
3. Note any control elements (valves, etc.)
4. Mention flow direction and connections
5. Highlight any gaps in the path

Response: Let me trace the flow path for you."""

def get_prompt_templates():
    """Get enhanced prompt templates for different query types."""
    return {
        'main': PromptTemplate(
            template=MAIN_QA_TEMPLATE,
            input_variables=["context", "graph_data", "chat_history", "question"]
        ),
        'equipment': PromptTemplate(
            template=EQUIPMENT_TEMPLATE,
            input_variables=["context", "graph_data", "question"]
        ),
        'flow': PromptTemplate(
            template=FLOW_TEMPLATE,
            input_variables=["context", "graph_data", "question"]
        )
    }