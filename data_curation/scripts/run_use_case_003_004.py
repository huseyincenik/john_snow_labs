
import time
import json
import urllib.request
import urllib.error
import sys
import shutil
from pathlib import Path

# API Endpoint (pointing to localhost:8000 exposed by Docker)
API_URL = "http://localhost:8000/api/v1"

def main():
    print("Starting Use Case Script (Client Mode)...")
    
    # 1. Start Process
    print("Sending process request...")
    req_data = {
        "doc_ids": ["jsl_p01_003_radiology_doc", "jsl_p01_004_clinical_doc"],
        "llm_provider": "openai"
    }
    
    req = urllib.request.Request(
        f"{API_URL}/process",
        data=json.dumps(req_data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            session_id = data['session_id']
            print(f"Session started: {session_id}")
    except urllib.error.URLError as e:
        print(f"Failed to connect to API: {e}")
        print("Ensure 'run_docker.sh' is running and port 8000 is accessible.")
        sys.exit(1)

    # 2. Poll Status
    print("Polling status...")
    status = "pending"
    while True:
        try:
            with urllib.request.urlopen(f"{API_URL}/status/{session_id}") as response:
                status_data = json.loads(response.read().decode('utf-8'))
                status = status_data['status']
                print(f"Status: {status} - {status_data.get('message', '')}")
                
                if status in ['completed', 'failed']:
                    if status == 'failed':
                        print("Processing failed!")
                        sys.exit(1)
                    break
        except Exception as e:
            print(f"Error polling status: {e}")
            
        time.sleep(2)

    # 3. Copy Artifacts
    print("Processing complete. Copying artifacts...")
    output_base = Path("data/output/use_case")
    # Clean existing
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Session output dir (mapped volume on host)
    session_dir = Path(f"data/output/{session_id}")
    
    if not session_dir.exists():
        print(f"Error: Session directory not found on host: {session_dir}")
        print("Note: This script assumes it is running on the host that has 'data' volume mounted.")
        sys.exit(1)

    # Define mappings (Source File Pattern -> Destination Filename)
    # We look for files in session_dir
    
    # 1. Tagger Result
    tagger_files = list(session_dir.glob("*stage_tagger*sorted.json"))
    if tagger_files:
        shutil.copy(tagger_files[0], output_base / "tagger_result.json")
        print(f"Copied {tagger_files[0].name} to tagger_result.json")
    
    # Locate intermediate directory
    # Intermediates might be in root (consolidated) or patient-specific folders
    # For Use Case 003/004, we know patient is "p01"
    
    root_intermediate = session_dir / "docetl_intermediate" / "clinical_registry"
    patient_intermediate = session_dir / "patients" / "p01" / "docetl_intermediate" / "clinical_registry"
    
    # Helper to copy from best source
    def copy_if_exists(filenames: list, dest_name: str):
        # Try root first (consolidated), then patient (individual)
        for fname in filenames:
            # Check root
            if (root_intermediate / fname).exists():
                shutil.copy(root_intermediate / fname, output_base / dest_name)
                print(f"Copied {fname} (from root) to {dest_name}")
                return
            # Check patient
            if (patient_intermediate / fname).exists():
                shutil.copy(patient_intermediate / fname, output_base / dest_name)
                print(f"Copied {fname} (from p01) to {dest_name}")
                return
        print(f"Warning: Could not find {filenames[0]}")

    # 2. Map Output
    copy_if_exists(["extract_clinical_fields.json"], "map_output.json")

    # 3. Normalize Output
    copy_if_exists(["normalize_extractions.json"], "normalize_extractions.json")
        
    # 4. Unnest Output
    copy_if_exists(["explode_field_records.json"], "explode_field_records.json")
        
    # 5. Resolve Output
    copy_if_exists(["resolve_patient_fields.json"], "resolve_output.json")

    # 6. Reduce Output (Intermediate)
    copy_if_exists(["reduce_patient_summary.json"], "reduce_patient_summary.json")
        
    # 7. Final Output (Consolidation)
    consolidation_files = list(session_dir.glob("*stage_consolidator*consolidation.json"))
    if consolidation_files:
        shutil.copy(consolidation_files[0], output_base / "final_output.json")
        print(f"Copied {consolidation_files[0].name} to final_output.json")

    print(f"\nAll artifacts copied to {output_base}")

if __name__ == "__main__":
    main()
