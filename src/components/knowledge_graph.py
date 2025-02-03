import logging
from neo4j import GraphDatabase
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

class KnowledgeGraph:
    """Handles P&ID Knowledge Graph operations in Neo4j"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_env()
        self._initialize_connection()
        
    def _load_env(self):
        """Load environment variables from .env file"""
        try:
            # Get the path to the root directory (where .env is located)
            current_dir = Path(__file__).resolve().parent.parent.parent
            env_path = current_dir / '.env'
            
            self.logger.info(f"Looking for .env file at: {env_path}")
            
            # Force reload environment variables
            load_dotenv(env_path, override=True)
            
            # Verify environment variables are loaded
            self.uri = os.getenv('NEO4J_URI')
            self.user = os.getenv('NEO4J_USER')
            self.password = os.getenv('NEO4J_PASSWORD')
            
            if not all([self.uri, self.user, self.password]):
                raise ValueError(
                    "Missing Neo4j credentials in .env file. "
                    f"URI: {'Set' if self.uri else 'Missing'}, "
                    f"USER: {'Set' if self.user else 'Missing'}, "
                    f"PASSWORD: {'Set' if self.password else 'Missing'}"
                )
            
            self.logger.info("Successfully loaded .env file")
                
        except Exception as e:
            self.logger.error(f"Error loading .env file: {str(e)}")
            raise
        
    def _initialize_connection(self):
        """Initialize Neo4j database connection using credentials"""
        try:
            self.logger.info(f"Attempting to connect using URI: {self.uri} and USER: {self.user}")
            
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Verify connectivity to ensure the server is reachable
            self.driver.verify_connectivity()
            self.logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def initialize_schema(self):
        """Create initial schema constraints and indexes"""
        try:
            with self.driver.session() as session:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (e:Equipment) 
                    REQUIRE e.id IS UNIQUE
                """)
                
                # Create indexes
                session.run("""
                    CREATE INDEX IF NOT EXISTS FOR (e:Equipment) 
                    ON (e.type)
                """)
                
            self.logger.info("Schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing schema: {str(e)}")
            raise

    def create_equipment(self, equipment_data: Dict[str, Any]):
        """Create an equipment node with properties"""
        query = """
        MERGE (e:Equipment {id: $id})
        SET e.type = $type,
            e.name = $name,
            e.description = $description,
            e += $properties
        RETURN e
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    id=equipment_data['id'],
                    type=equipment_data['type'],
                    name=equipment_data.get('name', ''),
                    description=equipment_data.get('description', ''),
                    properties={k: v for k, v in equipment_data.items() 
                              if k not in ['id', 'type', 'name', 'description']}
                )
                record = result.single()
                if record:
                    self.logger.info(f"Equipment node created: {record['e']}")
                    return record['e']
                else:
                    self.logger.warning("No equipment node returned for data: %s", equipment_data)
                    return None
                
        except Exception as e:
            self.logger.error(f"Error creating equipment: {str(e)}")
            raise

    def create_connection(self, from_id: str, to_id: str, properties: Dict[str, Any] = None):
        """Create a connection between two pieces of equipment"""
        if properties is None:
            properties = {}
            
        query = """
        MATCH (from:Equipment {id: $from_id})
        MATCH (to:Equipment {id: $to_id})
        MERGE (from)-[r:CONNECTS_TO]->(to)
        SET r += $properties
        RETURN r
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    from_id=from_id,
                    to_id=to_id,
                    properties=properties
                )
                record = result.single()
                if record:
                    self.logger.info(f"Connection created: {record['r']}")
                    return record['r']
                else:
                    self.logger.warning("No connection returned between %s and %s", from_id, to_id)
                    return None
                
        except Exception as e:
            self.logger.error(f"Error creating connection: {str(e)}")
            raise

    def get_equipment_by_type(self, equipment_type: str) -> List[Dict[str, Any]]:
        """Get all equipment of a specific type"""
        query = """
        MATCH (e:Equipment {type: $type})
        RETURN e.id as id, e.name as name, e.description as description,
               e.type as type, properties(e) as properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, type=equipment_type)
                equipment_list = [dict(record) for record in result]
                self.logger.info(f"Found {len(equipment_list)} equipment of type '{equipment_type}'")
                return equipment_list
                
        except Exception as e:
            self.logger.error(f"Error getting equipment by type: {str(e)}")
            raise

    def get_connected_equipment(self, equipment_id: str) -> List[Dict[str, Any]]:
        """Get all equipment connected to a specific piece of equipment"""
        query = """
        MATCH (e:Equipment {id: $id})-[r:CONNECTS_TO]-(connected:Equipment)
        RETURN connected.id as id, connected.type as type,
               type(r) as connection_type, properties(r) as properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, id=equipment_id)
                connected_list = [dict(record) for record in result]
                self.logger.info(f"Found {len(connected_list)} connected equipment for '{equipment_id}'")
                return connected_list
                
        except Exception as e:
            self.logger.error(f"Error getting connected equipment: {str(e)}")
            raise

    def get_flow_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Find flow path between two pieces of equipment"""
        query = """
        MATCH path = shortestPath(
            (start:Equipment {id: $start_id})-[:CONNECTS_TO*]-(end:Equipment {id: $end_id})
        )
        UNWIND nodes(path) as node
        RETURN collect(node.id) as path_ids, collect(node.type) as equipment_types
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, start_id=start_id, end_id=end_id)
                paths = [dict(record) for record in result]
                self.logger.info(f"Found path between {start_id} and {end_id}")
                return paths
                
        except Exception as e:
            self.logger.error(f"Error finding flow path: {str(e)}")
            raise
    
    def get_equipment_by_id(self, equipment_id: str) -> Dict[str, Any]:
        """Get equipment details by ID."""
        query = """
        MATCH (e:Equipment {id: $id})
        RETURN e.id as id, e.type as type, e.name as name, e.description as description
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, id=equipment_id)
                record = result.single()
                if record:
                    return {
                        'id': record['id'],
                        'type': record['type'],
                        'name': record['name'],
                        'description': record['description']
                    }
                self.logger.warning(f"No equipment found with ID: {equipment_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting equipment by ID: {str(e)}")
            raise

    def delete_equipment(self, equipment_id: str) -> bool:
        """Delete a piece of equipment and its connections"""
        query = """
        MATCH (e:Equipment {id: $id})
        DETACH DELETE e
        """
        try:
            with self.driver.session() as session:
                session.run(query, id=equipment_id)
                self.logger.info(f"Deleted equipment: {equipment_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting equipment: {str(e)}")
            raise

    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("Database cleared")
                
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}")
            raise

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j driver closed successfully")

if __name__ == "__main__":
    # Configure logging to print messages to the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    print("Starting KnowledgeGraph test...")
    
    kg = None
    try:
        kg = KnowledgeGraph()
        
        # Initialize schema
        print("Initializing schema...")
        kg.initialize_schema()
        
        # Create test equipment
        pump_data = {
            'id': 'P-101',
            'type': 'pump',
            'name': 'Feed Pump',
            'description': 'Main feed pump',
            'flow_rate': '100 L/min'
        }
        
        tank_data = {
            'id': 'T-101',
            'type': 'tank',
            'name': 'Storage Tank',
            'description': 'Raw material storage',
            'capacity': '1000L'
        }
        
        # Create equipment
        pump = kg.create_equipment(pump_data)
        tank = kg.create_equipment(tank_data)
        
        # Create connection
        connection = kg.create_connection(
            'T-101',
            'P-101',
            {'pipe_size': '3 inch', 'material': 'carbon steel'}
        )
        
        # Test queries
        pumps = kg.get_equipment_by_type('pump')
        print(f"Found pumps: {pumps}")
        
        connected = kg.get_connected_equipment('T-101')
        print(f"Connected to T-101: {connected}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if kg:
            kg.close()
        print("Test completed.")