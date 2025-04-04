import xml.etree.ElementTree as ET
import os
from pathlib import Path

def parse_xml_case(xml_path):
    """Parse annotated XML case file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract text content
    text_element = root.find('TEXT')
    case_text = text_element.text if text_element is not None else ""
    
    # Extract annotations
    annotations = {}
    tags = root.find('TAGS')
    
    if tags is not None:
        for tag in tags:
            tag_type = tag.tag
            
            annotation = {
                'text': tag.get('text', ''),
                'spans': tag.get('spans', '')
            }
            
            # Special handling for specific tags
            if tag_type == 'ESI':
                try:
                    annotation['esi_level'] = int(tag.get('ESI_LEVEL', 0))
                except:
                    annotation['esi_level'] = 0
            elif tag_type == 'QA':
                annotation['relevance'] = tag.get('relevance', '')
                annotation['comment'] = tag.get('comment', '')
            
            if tag_type not in annotations:
                annotations[tag_type] = []
            annotations[tag_type].append(annotation)
    
    return {
        'case_text': case_text,
        'annotations': annotations
    }

def get_xml_files(directory, limit=None):
    """Get list of XML files from directory"""
    # Change this line to match your actual file extension
    xml_files = list(Path(directory).glob('*.txt.xml'))
    if limit:
        xml_files = xml_files[:limit]
    return xml_files

def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs('output/confidence_analysis', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)