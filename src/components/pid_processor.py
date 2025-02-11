"""
Enhanced PID Processor Component with Gemini Pro Vision support
"""

import os
import logging
import base64
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
from pdf2image import convert_from_path
from datetime import datetime


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
        
        # Initialize Gemini
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.logger.warning("GOOGLE_API_KEY not found, Gemini functionality will be disabled")
            self.gemini_model = None
        
        # Create debug output directory
        current_dir = Path.cwd()
        self.debug_dir = current_dir / "debug_output"
        self.debug_dir.mkdir(exist_ok=True)
        self.logger.info(f"Debug output directory: {self.debug_dir}")

    def process_file(self, file_path: str, use_gemini: bool = False) -> Dict:
        """Process a P&ID file using either OpenAI Vision or Gemini Pro Vision."""
        self.logger.info(f"=== Starting P&ID Processing ===")
        self.logger.info(f"Using model: {'Gemini' if use_gemini else 'OpenAI'}")
        self.logger.info(f"Input file: {file_path}")
        
        try:
            # Handle PDF files
            if file_path.lower().endswith('.pdf'):
                return self._process_pdf(file_path, use_gemini)
            else:
                # Process image files as before
                if use_gemini and self.gemini_model:
                    return self._process_with_gemini(file_path)
                else:
                    return self._process_with_openai(file_path)
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    
    
    def _process_pdf(self, pdf_path: str, use_gemini: bool = False) -> Dict:
        """Process PDF file by converting to images first."""
        try:
            self.logger.info(f"Converting PDF to images: {pdf_path}")
            
            # Get absolute path to poppler
            current_dir = Path(__file__).parent.parent.parent   # Go up three levels to reach project root
            poppler_path = str(current_dir / "tools" / "poppler-24.08.0" / "Library" / "bin")
            
            self.logger.info(f"Using poppler path: {poppler_path}")
            
            if not os.path.exists(poppler_path):
                raise ValueError(f"Poppler not found at {poppler_path}")
                
            # Convert PDF to images with explicit poppler path
            try:
                images = convert_from_path(
                    pdf_path,
                    poppler_path=poppler_path,
                    dpi=200,  # Adjust DPI as needed
                    fmt='png'
                )
                self.logger.info(f"Converted PDF to {len(images)} pages")
            except Exception as e:
                self.logger.error(f"PDF conversion error: {str(e)}")
                raise ValueError(f"Failed to convert PDF: {str(e)}")
            
            all_equipment = []
            all_connections = []
            
            # Process each page
            for i, image in enumerate(images):
                self.logger.info(f"Processing page {i+1}/{len(images)}")
                
                # Save temporary image
                temp_img_path = os.path.join(self.debug_dir, f"temp_page_{i+1}.png")
                image.save(temp_img_path)
                
                # Process the image
                try:
                    if use_gemini and self.gemini_model:
                        result = self._process_with_gemini(temp_img_path)
                    else:
                        result = self._process_with_openai(temp_img_path)
                        
                    if result['status'] == 'success':
                        all_equipment.extend(result['equipment'])
                        all_connections.extend(result['connections'])
                    
                finally:
                    # Clean up temporary image
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
            
            # Save combined results
            if all_equipment:
                self._save_debug_info(all_equipment, all_connections)
                if self.knowledge_graph:
                    self._store_in_knowledge_graph(all_equipment, all_connections)
            
            return {
                'status': 'success',
                'equipment': all_equipment,
                'connections': all_connections,
                'file_path': pdf_path,
                'model': 'gemini' if use_gemini else 'openai',
                'pages_processed': len(images)
            }
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': f"PDF processing error: {str(e)}"}
    
    def _process_with_openai(self, file_path: str) -> Dict:
        """Process using OpenAI Vision API."""
        try:
            # Load and encode image
            image_base64 = self._encode_image(file_path)
            if not image_base64:
                return {'status': 'error', 'message': 'Failed to load image file'}
            
            # Existing OpenAI processing code...
            equipment_list = self._analyze_with_openai(image_base64)
            
            # Process connections
            connections = self._detect_connections(equipment_list)
            
            return {
                'status': 'success',
                'equipment': equipment_list,
                'connections': connections,
                'file_path': file_path
            }
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI processing: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _process_with_gemini(self, file_path: str) -> Dict:
        """Process using Gemini Pro Vision."""
        try:
            # Load image for Gemini
            image = Image.open(file_path)
            
            # Create prompt for Gemini
            prompt = """Analyze this P&ID diagram and identify equipment with exact formatting:

            For each piece of equipment, provide:
            - Type: [exact type - tank, pump, or valve]
            - ID: [exact ID from diagram]

            Format each piece of equipment exactly as shown above, with a blank line between equipment.
            Do not include any other information or explanations.
            Example format:
            - Type: pump
            ID: P-101

            - Type: valve
            ID: V-201"""
            
            # Get response from Gemini
            response = self.gemini_model.generate_content([prompt, image])
            
            # Parse Gemini response
            equipment_list = self._parse_gemini_response(response.text)
            
            # Process connections if equipment was found
            connections = []
            if equipment_list:
                connections = self._detect_connections(equipment_list)
                
                # Save debug info
                self._save_debug_info(equipment_list, connections)
                
                # Store in knowledge graph if available
                if self.knowledge_graph:
                    self._store_in_knowledge_graph(equipment_list, connections)
            
            return {
                'status': 'success',
                'equipment': equipment_list,
                'connections': connections,
                'file_path': file_path,
                'model': 'gemini'
            }
            
        except Exception as e:
            self.logger.error(f"Error in Gemini processing: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's response into structured equipment list."""
        equipment_list = []
        current_equipment = {}
        
        try:
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Start new equipment on type definition
                if 'Type:' in line or line.startswith('- Type:'):
                    if current_equipment and 'type' in current_equipment and 'id' in current_equipment:
                        equipment_list.append(current_equipment)
                    current_equipment = {}
                    type_value = line.split('Type:')[1].strip().lower()
                    current_equipment['type'] = type_value
                    
                elif 'ID:' in line:
                    id_value = line.split('ID:')[1].strip()
                    current_equipment['id'] = id_value
                    
                elif 'Connected to:' in line:
                    connections = line.split('Connected to:')[1].strip()
                    # Handle both comma-separated and list formats
                    if '[' in connections:
                        connections = connections.strip('[]')
                    current_equipment['connections'] = [
                        conn.strip() for conn in connections.split(',') if conn.strip()
                    ]
            
            # Add the last equipment if it's complete
            if current_equipment and 'type' in current_equipment and 'id' in current_equipment:
                equipment_list.append(current_equipment)
            
            # Ensure all equipment has required fields
            for eq in equipment_list:
                if 'id' not in eq:
                    eq['id'] = f"unknown_{eq.get('type', 'equipment')}"
                if 'type' not in eq:
                    eq['type'] = 'unknown'
            
            self.logger.info(f"Parsed equipment list: {equipment_list}")
            return equipment_list
                
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}")
            # Return empty list on error
            return []

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
            self.logger.info(f"Saving debug info to {debug_file}")
            
            with open(debug_file, 'w') as f:
                f.write(f"Detection Time: {datetime.now()}\n")
                f.write("\nDetected Equipment:\n")
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
            
            self.logger.info(f"Storing {len(equipment_list)} equipment in knowledge graph...")
            # Store equipment
            for equipment in equipment_list:
                try:
                    equipment_data = {
                        'id': equipment['id'],
                        'type': equipment['type'],
                        'name': equipment['id'],
                        'description': f"Detected {equipment['type']}"
                    }
                    self.knowledge_graph.create_equipment(equipment_data)
                    self.logger.info(f"Stored equipment: {equipment['id']}")
                except Exception as e:
                    self.logger.error(f"Error storing equipment {equipment['id']}: {str(e)}")
            
            self.logger.info(f"Storing {len(connections)} connections in knowledge graph...")
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