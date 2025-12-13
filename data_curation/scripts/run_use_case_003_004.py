
import sys
import os
import asyncio
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.models.schemas import DocumentMetadata
from src.pipeline.tagger import Tagger
from src.pipeline.docetl_runner import DocETLPipelineRunner
from src.utils.ontology import OntologyLoader
from config.settings import settings

async def main():
    print("Starting Use Case Script for Documents 003 and 004...")
    
    # 1. Setup paths
    input_dir = Path("input_patient_docs")
    # Output base is relative to where we run. 
    # The user asked for "scripts" folder for the script, but run it to create "output/use_case"
    # If script is in scripts/, running it from project root means output/use_case is data_curation/output/use_case
    
    output_base = Path("data/output/use_case") 
    output_base.mkdir(parents=True, exist_ok=True)
    
    session_id = "use_case_run"
    
    # 2. Define Documents (Manually loading metadata based on file content)
    
    def parse_doc(filename):
        path = input_dir / filename
        if not path.exists():
            print(f"Error: {path} does not exist.")
            return None
            
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        parts = full_text.split("---", 1)
        header = parts[0]
        body = parts[1].strip() if len(parts) > 1 else ""
        
        metadata = {}
        for line in header.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip().lower()] = val.strip()
                
        return DocumentMetadata(
            patient_id=metadata.get("patient id", "p01"),
            doc_id=metadata.get("doc id", "unknown"),
            doc_type=metadata.get("doc type", "unknown"),
            doc_date=metadata.get("date", None),
            filename=str(path),
            content=body 
        )
        
    doc3 = parse_doc("jsl_p01_003_radiology_doc.txt")
    doc4 = parse_doc("jsl_p01_004_clinical_doc.txt")
    
    if not doc3 or not doc4:
        print("Failed to load documents.")
        return

    print(f"Loaded {doc3.doc_id} and {doc4.doc_id}")
    
    # 3. Run Tagger
    print("Running Tagger...")
    # Initialize Tagger with OpenAI provider logic (default in tagger if not specified, 
    # but we want to ensure settings.openrouter_model_openai is used if needed)
    # Tagger uses get_llm_provider() which uses settings.
    tagger = Tagger(provider_label="openai") 
    tagger_result, sorted_docs = await tagger.tag_documents([doc3, doc4], session_id)
    
    # Save Tagger Result
    tagger_json_path = output_base / "tagger_result.json"
    with open(tagger_json_path, "w", encoding="utf-8") as f:
        f.write(tagger_result.model_dump_json(indent=2))
    print(f"Tagger output saved to {tagger_json_path}")
    
    # 4. Run DocETL Pipeline
    print("Running DocETL Pipeline...")
    ontology = OntologyLoader()
    model_name = settings.openrouter_model_openai
    print(f"Using model: {model_name}")
    
    runner = DocETLPipelineRunner(ontology=ontology, model_name=model_name)
    
    # Run the pipeline
    artifacts = runner.run_pipeline(sorted_docs, session_id)
    
    # 5. Copy artifacts
    print("Copying artifacts to use_case folder...")
    
    # Map Output
    if artifacts.map_output_path and artifacts.map_output_path.exists():
        shutil.copy(artifacts.map_output_path, output_base / "map_output.json")
        print("Copied Map Output to map_output.json")
        
    # Resolve Output
    if artifacts.resolve_output_path and artifacts.resolve_output_path.exists():
        shutil.copy(artifacts.resolve_output_path, output_base / "resolve_output.json")
        print("Copied Resolve Output to resolve_output.json")
        
    # Final Output (Reduce)
    if artifacts.patient_output_path and artifacts.patient_output_path.exists():
        shutil.copy(artifacts.patient_output_path, output_base / "final_output.json")
        print("Copied Final Output to final_output.json")
        
    # Copy intermediate files (like Normalize, Unnest if simple copy didn't catch them)
    # artifacts.map_output_path is presumably extract_clinical_fields.json
    # docetl_intermediate/clinical_registry/ also has unnest/normalize usually?
    # Let's inspect the intermediate directory
    
    intermediate_dir = Path(settings.output_dir) / session_id / "docetl_intermediate" / "clinical_registry"
    if intermediate_dir.exists():
        for file in intermediate_dir.glob("*.json"):
            target = output_base / file.name
            if not target.exists(): # Don't overwrite what we already copied if we renamed it
                shutil.copy(file, target)
                print(f"Copied additional intermediate file: {file.name}")
            
    print(f"Execution complete. All artifacts in {output_base}")

if __name__ == "__main__":
    asyncio.run(main())
