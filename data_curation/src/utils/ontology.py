"""Ontology loader for cancer registry fields."""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class OntologyLoader:
    """Loads and manages cancer registry field ontology."""
    
    def __init__(self, ontology_path: Optional[Path] = None):
        if ontology_path is None:
            ontology_path = Path("data/ontology/cancer_registry_fields.yaml")
        self.ontology_path = ontology_path
        self.ontology: Dict[str, Any] = {}
        self._load_ontology()
    
    def _load_ontology(self):
        """Load ontology from YAML file."""
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            self.ontology = yaml.safe_load(f)
    
    def get_all_fields(self) -> List[Dict[str, Any]]:
        """Get all fields from all domains."""
        fields = []
        for domain_name, domain_data in self.ontology.get("domains", {}).items():
            domain_fields = domain_data.get("fields", [])
            for field in domain_fields:
                field["domain"] = domain_name
                fields.append(field)
        return fields
    
    def get_fields_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get fields for a specific domain."""
        domain_data = self.ontology.get("domains", {}).get(domain, {})
        return domain_data.get("fields", [])
    
    def get_field_definition(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get definition for a specific field."""
        for field in self.get_all_fields():
            if field.get("name") == field_name:
                return field
        return None
    
    def get_extraction_instructions(self) -> str:
        """Generate extraction instructions from ontology."""
        instructions = []
        instructions.append("# Cancer Registry Field Extraction Instructions\n")
        instructions.append("Extract the following fields from clinical documents:\n")
        
        for domain_name, domain_data in self.ontology.get("domains", {}).items():
            instructions.append(f"\n## {domain_name.upper().replace('_', ' ')}")
            profile = domain_data.get("profile", "")
            if profile:
                instructions.append(f"Profile: {profile}\n")
            
            for field in domain_data.get("fields", []):
                name = field.get("name", "")
                description = field.get("description", "")
                field_instructions = field.get("instructions", "")
                data_type = field.get("data_type", "string")
                
                instructions.append(f"\n### {name}")
                instructions.append(f"**Description:** {description}")
                instructions.append(f"**Data Type:** {data_type}")
                if field_instructions:
                    instructions.append(f"**Instructions:** {field_instructions}")
        
        return "\n".join(instructions)

