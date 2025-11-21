"""
End-to-end demo runner for the Data Curation Service.

This script:
1. Uploads sample documents.
2. Triggers processing for provided patient IDs.
3. Polls status until completion.
4. Persists raw API responses, extracted outputs, and a log file
   under demo_runs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import httpx


LOGGER = logging.getLogger("e2e-demo")


def setup_run_directory(base_dir: Path) -> Path:
    """Create a timestamped directory to store demo artifacts."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"demo_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logging(log_path: Path) -> None:
    """Configure logging to both console and file."""
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)


async def upload_documents(
    client: httpx.AsyncClient,
    base_url: str,
    files: Sequence[Path],
) -> dict:
    """Upload documents using the /upload endpoint."""
    LOGGER.info("Uploading %d documents", len(files))

    multipart_files = [
        (
            "files",
            (file.name, file.read_bytes(), "text/plain"),
        )
        for file in files
    ]

    response = await client.post(
        f"{base_url}/api/v1/upload",
        files=multipart_files,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    LOGGER.info("Upload response: %s", data["message"])
    return data


async def start_processing(
    client: httpx.AsyncClient,
    base_url: str,
    patient_ids: Sequence[str],
    process_all: bool,
    llm_provider: str,
    llm_model: Optional[str] = None,
    max_documents: Optional[int] = None,
) -> dict:
    """Trigger the /process endpoint."""
    payload = {
        "process_all": process_all,
        "llm_provider": llm_provider,
    }
    if patient_ids:
        payload["patient_ids"] = list(patient_ids)
    if llm_model:
        payload["llm_model"] = llm_model
    if max_documents is not None and max_documents > 0:
        payload["max_documents"] = max_documents
    LOGGER.info(
        "Starting processing with payload: %s",
        json.dumps(payload, indent=2),
    )
    response = await client.post(
        f"{base_url}/api/v1/process",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    LOGGER.info("Process started: session_id=%s", data["session_id"])
    return data


async def poll_status(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    interval: float = 5.0,
    timeout: float | None = 300.0,
    max_errors: int = 5,
) -> dict:
    """Poll the /status/{session_id} endpoint until completion or timeout."""
    LOGGER.info("Polling status for session_id=%s", session_id)
    elapsed = 0.0
    consecutive_errors = 0
    unlimited_timeout = timeout is None or timeout <= 0 or timeout == float("inf")

    while True:
        try:
            response = await client.get(
                f"{base_url}/api/v1/status/{session_id}",
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            LOGGER.info("Status: %s - %s", data["status"], data["message"])
            consecutive_errors = 0

            if data["status"] in {"completed", "failed"}:
                return data
        except httpx.RequestError as exc:
            consecutive_errors += 1
            LOGGER.warning(
                "Status polling error (%s/%s): %s",
                consecutive_errors,
                max_errors,
                exc,
            )
            if consecutive_errors >= max_errors:
                raise TimeoutError(
                    f"Status polling failed repeatedly for session " f"{session_id}"
                ) from exc

        await asyncio.sleep(interval)
        elapsed += interval

        if not unlimited_timeout and elapsed >= timeout:
            raise TimeoutError(
                f"Processing did not finish within {timeout} seconds " f"(session_id={session_id})"
            )


async def poll_status_with_extensions(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    interval: float,
    timeout_plan: list[float],
    max_errors: int = 5,
) -> dict:
    """Sequentially retry poll_status with increasing timeouts."""
    last_exc: TimeoutError | None = None

    positive_timeouts = [t for t in timeout_plan if t is not None and t > 0 and t != float("inf")]
    if not positive_timeouts:
        return await poll_status(
            client=client,
            base_url=base_url,
            session_id=session_id,
            interval=interval,
            timeout=None,
            max_errors=max_errors,
        )

    for idx, timeout in enumerate(timeout_plan):
        if timeout is None or timeout <= 0 or timeout == float("inf"):
            return await poll_status(
                client=client,
                base_url=base_url,
                session_id=session_id,
                interval=interval,
                timeout=None,
                max_errors=max_errors,
            )
        try:
            LOGGER.info(
                "Polling session_id=%s with timeout %.0f seconds (step %d/%d)",
                session_id,
                timeout,
                idx + 1,
                len(timeout_plan),
            )
            return await poll_status(
                client=client,
                base_url=base_url,
                session_id=session_id,
                interval=interval,
                timeout=timeout,
                max_errors=max_errors,
            )
        except TimeoutError as exc:
            last_exc = exc
            if idx < len(timeout_plan) - 1:
                LOGGER.warning(
                    "Session %s still processing after %.0fs; extending timeout to %.0fs",
                    session_id,
                    timeout,
                    timeout_plan[idx + 1],
                )
            else:
                LOGGER.error(
                    "Session %s exceeded maximum timeout plan (%s)",
                    session_id,
                    timeout_plan,
                )
    if last_exc:
        raise last_exc
    raise TimeoutError(f"Polling failed for session_id={session_id}")


def save_json(data: dict, path: Path) -> None:
    """Persist dictionary as JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


async def run_demo(args: argparse.Namespace) -> None:
    """Execute the complete demo workflow."""
    base_dir = Path(args.output_dir)
    run_dir = setup_run_directory(base_dir)
    configure_logging(run_dir / "demo.log")
    LOGGER.info("Artifacts will be stored under %s", run_dir)

    sample_dir = Path(args.sample_dir)
    sample_files = sorted(sample_dir.glob("*.txt"))
    if not sample_files:
        raise FileNotFoundError(
            f"No .txt files found in {sample_dir}. " "Provide valid sample documents."
        )
    provider_models = parse_provider_models(args.provider_models)

    def resolve_doc_slice(provider_name: str) -> tuple[list[Path], Optional[int]]:
        """Return the document subset and max-doc limit for a provider."""
        provider_lower = provider_name.lower()
        provider_limits = {
            "qwen": args.qwen_doc_limit,
            "openai": args.openai_doc_limit,
        }
        limit = provider_limits.get(provider_lower, args.num_docs)
        if limit is None or limit <= 0:
            return sample_files, None
        return sample_files[: min(limit, len(sample_files))], limit

    async with httpx.AsyncClient() as client:
        for index, provider in enumerate(args.llm_providers, start=1):
            provider_slug = provider.lower().replace(" ", "_")
            provider_dir = run_dir / f"{index:02d}_{provider_slug}"
            provider_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("=== Provider %s ===", provider)

            provider_sample_slice, provider_doc_limit = resolve_doc_slice(provider)
            LOGGER.info(
                "Provider %s will use %d document(s)",
                provider,
                len(provider_sample_slice),
            )

            if args.upload:
                upload_response = await upload_documents(
                    client, args.base_url, provider_sample_slice
                )
                save_json(
                    upload_response,
                    provider_dir / "01_upload_response.json",
                )

            process_response = await start_processing(
                client=client,
                base_url=args.base_url,
                patient_ids=args.patient_ids,
                process_all=args.process_all,
                llm_provider=provider,
                llm_model=provider_models.get(provider.lower()),
                max_documents=provider_doc_limit,
            )
            save_json(
                process_response,
                provider_dir / "02_process_response.json",
            )

            session_id = process_response["session_id"]
            if provider.lower() == "qwen":
                timeout_plan = [
                    max(args.poll_timeout, args.qwen_timeout),
                    max(args.poll_timeout, 3600.0, args.qwen_timeout),
                    max(args.poll_timeout, 4800.0, args.qwen_timeout),
                ]
                status_response = await poll_status_with_extensions(
                    client=client,
                    base_url=args.base_url,
                    session_id=session_id,
                    interval=args.poll_interval,
                    timeout_plan=timeout_plan,
                )
            else:
                status_response = await poll_status(
                    client=client,
                    base_url=args.base_url,
                    session_id=session_id,
                    interval=args.poll_interval,
                    timeout=args.poll_timeout,
                )
            save_json(
                status_response,
                provider_dir / "03_status_final.json",
            )

            # Persist extraction & consolidation payloads if present
            tagger = status_response.get("tagger_result")
            if tagger:
                save_json(tagger, provider_dir / "tagger_result.json")
            extraction = status_response.get("extraction_result")
            if extraction:
                save_json(extraction, provider_dir / "extraction_result.json")
            consolidation = status_response.get("consolidation_result")
            if consolidation:
                save_json(
                    consolidation,
                    provider_dir / "consolidation_result.json",
                )

            LOGGER.info(
                "Provider %s completed. Session ID: %s (artifacts: %s)",
                provider,
                session_id,
                provider_dir,
            )


def parse_provider_models(raw_pairs: Sequence[str]) -> dict[str, str]:
    """Parse CLI-supplied provider=model mappings."""
    mapping: dict[str, str] = {}
    for pair in raw_pairs:
        if "=" not in pair:
            continue
        provider, model = pair.split("=", 1)
        mapping[provider.strip().lower()] = model.strip()
    return mapping


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run an end-to-end DocETL pipeline demo")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="FastAPI hizmetinin temel URL'si",
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        default=["p01"],
        help="İşlenecek hasta kimlikleri",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Tüm belgeleri işle",
    )
    parser.add_argument(
        "--sample-dir",
        default="input_patient_docs",
        help="Yüklenecek örnek belgelerin klasörü",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=0,
        help="Varsayılan belge sayısı (0: tüm belgeler)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="/upload uç noktasını da çalıştır",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_runs",
        help="Demo çıktı klasörü",
    )
    parser.add_argument(
        "--llm-providers",
        nargs="+",
        default=["qwen", "openai"],
        help="Sırayla test edilecek LLM sağlayıcıları",
    )
    parser.add_argument(
        "--provider-models",
        action="append",
        default=[],
        metavar="PROVIDER=MODEL",
        help="Sağlayıcıya özel model eşlemesi (örn. openai=gpt-4o-mini)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Durum sorgulama aralığı (saniye)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=3600.0,
        help="İşlemin tamamlanması için maksimum süre (saniye)",
    )
    parser.add_argument(
        "--qwen-timeout",
        type=float,
        default=3600.0,
        help="Qwen sağlayıcısı için minimum bekleme süresi (saniye)",
    )
    parser.add_argument(
        "--qwen-doc-limit",
        type=int,
        default=1,
        help="Qwen demosunda işlenecek belge sayısı (1 önerilir)",
    )
    parser.add_argument(
        "--openai-doc-limit",
        type=int,
        default=0,
        help="OpenAI demosunda işlenecek belge sayısı (0: tüm belgeler)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
