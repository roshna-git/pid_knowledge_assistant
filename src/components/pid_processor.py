"""
PID Processor Component using OpenAI Vision API
"""

import os
import logging
import base64
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI

class PIDProcessor:
    def __init__(self, knowledge_graph=None):
        self.logger = logging.getLogger(__name__)
        self.knowledge_graph = knowledge_graph
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg']
        
        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Create debug output directory
        current_dir = Path.cwd()
        self.debug_dir = current_dir / "debug_output"
        self.debug_dir.mkdir(exist_ok=True)
        self.logger.info(f"Debug output directory: {self.debug_dir}")

    def process_file(self, file_path: str) -> Dict:
        """Process a P&ID file using OpenAI Vision API."""
        self.logger.info(f"=== Starting P&ID Processing with OpenAI ===")
        self.logger.info(f"Input file: {file_path}")
        
        try:
            # Load and encode image
            image_base64 = self._encode_image(file_path)
            if not image_base64:
                return {'status': 'error', 'message': 'Failed to load image file'}
            
            # Analyze image with OpenAI Vision
            equipment_list = self._analyze_with_vision_api(image_base64)
            
            # Process connections
            connections = []
            if len(equipment_list) > 1:
                connections = self._detect_connections(equipment_list)
            
            # Save debug visualization
            self._save_debug_info(equipment_list, connections)
            
            # Store in knowledge graph if available
            if self.knowledge_graph and equipment_list:
                self._store_in_knowledge_graph(equipment_list, connections)
            
            return {
                'status': 'success',
                'equipment': equipment_list,
                'connections': connections,
                'file_path': file_path
            }
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {str(e)}")
            return None

    def _analyze_with_vision_api(self, image_base64: str) -> List[Dict]:
        """Analyze P&ID using OpenAI Vision API."""
        try:
            system_prompt = """You are a P&ID expert. Analyze this image and identify:
            1. Tanks (rectangular shapes, usually labeled T-xxx)
            2. Pumps (circular shapes with cross, usually labeled P-xxx)
            3. Valves (small square/diamond shapes, usually labeled V-xxx)
            
            For each equipment, provide only:
            - Type (tank, pump, or valve)
            - Equipment ID (as shown in image)
            
            Format each equipment as:
            - Type: [type]
              ID: [id]"""

            self.logger.info("Sending request to OpenAI Vision API...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            raw_response = response.choices[0].message.content
            self.logger.info(f"Raw API response: {raw_response}")
            
            # Parse the response
            equipment_list = []
            current_equipment = {}
            
            for line in raw_response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('- Type:'):
                    if current_equipment:
                        equipment_list.append(current_equipment)
                    current_equipment = {}
                    current_equipment['type'] = line.split(':')[1].strip().lower()
                elif line.startswith('ID:'):
                    current_equipment['id'] = line.split(':')[1].strip()
            
            # Add the last equipment if exists
            if current_equipment and 'type' in current_equipment and 'id' in current_equipment:
                equipment_list.append(current_equipment)
            
            self.logger.info(f"Detected equipment: {equipment_list}")
            return equipment_list

        except Exception as e:
            self.logger.error(f"Error in Vision API analysis: {str(e)}", exc_info=True)
            raise

    def _detect_connections(self, equipment_list: List[Dict]) -> List[Dict]:
        """Create connections based on equipment order."""
        connections = []
        for i in range(len(equipment_list) - 1):
            connections.append({
                'from_id': equipment_list[i]['id'],
                'to_id': equipment_list[i + 1]['id'],
                'type': 'flow'
            })
        return connections

    def _save_debug_info(self, equipment_list: List[Dict], connections: List[Dict]):
        """Save detection results to debug file."""
        try:
            debug_file = self.debug_dir / "detection_results.txt"
            with open(debug_file, 'w') as f:
                f.write("Detected Equipment:\n")
                for eq in equipment_list:
                    eq_type = eq.get('type', 'unknown')
                    eq_id = eq.get('id', 'unknown')
                    f.write(f"- {eq_type}: {eq_id}\n")
                
                f.write("\nDetected Connections:\n")
                for conn in connections:
                    from_id = conn.get('from_id', 'unknown')
                    to_id = conn.get('to_id', 'unknown')
                    f.write(f"- {from_id} -> {to_id}\n")
            
            self.logger.info(f"Saved detection results to {debug_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug info: {str(e)}", exc_info=True)

    def _store_in_knowledge_graph(self, equipment_list: List[Dict], connections: List[Dict]):
        """Store detected equipment and connections in the knowledge graph."""
        try:
            if not self.knowledge_graph:
                self.logger.warning("No knowledge graph available for storage")
                return
            
            self.logger.info("Storing equipment in knowledge graph...")
            # Store equipment
            for equipment in equipment_list:
                try:
                    self.knowledge_graph.create_equipment({
                        'id': equipment['id'],
                        'type': equipment['type'],
                        'name': equipment['id'],
                        'description': f"Detected {equipment['type']}"
                    })
                    self.logger.info(f"Stored equipment: {equipment['id']}")
                except Exception as e:
                    self.logger.error(f"Error storing equipment {equipment['id']}: {str(e)}")
            
            self.logger.info("Storing connections in knowledge graph...")
            # Store connections
            for conn in connections:
                try:
                    self.knowledge_graph.create_connection(
                        conn['from_id'],
                        conn['to_id'],
                        {'type': 'flow'}
                    )
                    self.logger.info(f"Stored connection: {conn['from_id']} -> {conn['to_id']}")
                except Exception as e:
                    self.logger.error(f"Error storing connection: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error storing in knowledge graph: {str(e)}", exc_info=True)