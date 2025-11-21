"""Storage utilities for saving results."""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from config.settings import settings
from src.models.schemas import ExtractionResult, ConsolidationResult, TaggerResult


class StorageManager:
    """Manages storage of extraction and consolidation results."""
    
    def __init__(self, storage_type: Optional[str] = None):
        self.storage_type = storage_type or settings.storage_type
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_tagger(
        self,
        result: TaggerResult,
        session_id: str,
    ) -> Path:
        """Save tagger result (sorted documents) to storage."""
        if self.storage_type == "json":
            return self._save_json(result, session_id, "sorted")
        elif self.storage_type == "postgres":
            return self._save_postgres(result, session_id, "sorted")
        raise ValueError(f"Unknown storage type: {self.storage_type}")
    
    def save_extraction(
        self,
        result: ExtractionResult,
        session_id: str,
    ) -> Path:
        """Save extraction result to storage."""
        if self.storage_type == "json":
            return self._save_json(result, session_id, "extraction")
        elif self.storage_type == "postgres":
            return self._save_postgres(result, session_id, "extraction")
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
    
    def save_consolidation(
        self,
        result: ConsolidationResult,
        session_id: str,
    ) -> Path:
        """Save consolidation result to storage."""
        if self.storage_type == "json":
            return self._save_json(result, session_id, "consolidation")
        elif self.storage_type == "postgres":
            return self._save_postgres(result, session_id, "consolidation")
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
    
    def _save_json(
        self,
        result: TaggerResult | ExtractionResult | ConsolidationResult,
        session_id: str,
        result_type: str,
    ) -> Path:
        """Save result as JSON file."""
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"stage_{result.stage}_{session_id}_{result_type}.json"
        filepath = session_dir / filename
        
        # Convert to dict with datetime serialization
        data = result.model_dump(mode="json")
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_postgres(
        self,
        result: TaggerResult | ExtractionResult | ConsolidationResult,
        session_id: str,
        result_type: str,
    ) -> Path:
        """Save result to PostgreSQL (placeholder for future implementation)."""
        # TODO: Implement PostgreSQL storage
        # For now, fallback to JSON
        return self._save_json(result, session_id, result_type)
    
    def load_tagger(self, session_id: str) -> Optional[TaggerResult]:
        """Load tagger result from storage."""
        if self.storage_type == "json":
            return self._load_json(session_id, "sorted", TaggerResult)
        raise NotImplementedError("PostgreSQL loading not yet implemented")
    
    def load_extraction(self, session_id: str) -> Optional[ExtractionResult]:
        """Load extraction result from storage."""
        if self.storage_type == "json":
            return self._load_json(session_id, "extraction", ExtractionResult)
        else:
            raise NotImplementedError("PostgreSQL loading not yet implemented")
    
    def load_consolidation(self, session_id: str) -> Optional[ConsolidationResult]:
        """Load consolidation result from storage."""
        if self.storage_type == "json":
            return self._load_json(session_id, "consolidation", ConsolidationResult)
        else:
            raise NotImplementedError("PostgreSQL loading not yet implemented")
    
    def _load_json(
        self,
        session_id: str,
        result_type: str,
        model_class: type,
    ) -> Optional[Any]:
        """Load result from JSON file."""
        session_dir = self.output_dir / session_id
        
        pattern = f"stage_*_{session_id}_{result_type}.json"
        files = list(session_dir.glob(pattern))
        
        if not files:
            return None
        
        filepath = max(files, key=lambda p: p.stat().st_mtime)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return model_class(**data)

