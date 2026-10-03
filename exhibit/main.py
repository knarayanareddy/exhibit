from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import uvicorn

from .api import EvidenceRepository, Settings, create_app
from .evaluator import heuristic_evaluate_trace
from .models import CONSTITUTIONAL_RULE
from .spans import ingest_jsonl


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exhibit",
        description=(
            "Compile immutable EU AI Act engineering evidence. "
            "This tool does not perform a conformity assessment."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser(
        "serve",
        help="Run the FastAPI compliance dashboard",
    )
    serve.add_argument("--host", default=os.getenv("EXHIBIT_HOST", "127.0.0.1"))
    serve.add_argument(
        "--port",
        type=_port,
        default=_port(os.getenv("EXHIBIT_PORT", "8000")),
    )
    serve.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.getenv("EXHIBIT_LOG_LEVEL", "info"),
    )
    serve.add_argument(
        "--db",
        type=Path,
        default=Path(os.getenv("EXHIBIT_DB_PATH", str(Path.cwd() / "receipts.db"))),
    )
    serve.add_argument(
        "--artifacts",
        type=Path,
        default=Path(os.getenv("EXHIBIT_ARTIFACT_DIR", str(Path.cwd() / "artifacts"))),
    )

    demo = subcommands.add_parser(
        "demo",
        help="Ingest bundled JSONL fixtures and optionally start the dashboard",
    )
    demo.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures" / "traces",
    )
    demo.add_argument("--no-serve", action="store_true")
    demo.add_argument("--host", default=os.getenv("EXHIBIT_HOST", "127.0.0.1"))
    demo.add_argument(
        "--port",
        type=_port,
        default=_port(os.getenv("EXHIBIT_PORT", "8000")),
    )
    demo.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.getenv("EXHIBIT_LOG_LEVEL", "info"),
    )
    demo.add_argument(
        "--db",
        type=Path,
        default=Path(os.getenv("EXHIBIT_DB_PATH", str(Path.cwd() / "demo-receipts.db"))),
    )
    demo.add_argument(
        "--artifacts",
        type=Path,
        default=Path(os.getenv("EXHIBIT_ARTIFACT_DIR", str(Path.cwd() / "demo-artifacts"))),
    )
    return parser


async def _ingest_demo_fixtures(application: Any, fixtures: Path) -> list[dict[str, object]]:
    repository: EvidenceRepository = application.state.repository
    settings: Settings = application.state.settings
    if not fixtures.is_dir():
        raise FileNotFoundError(f"trace fixture directory does not exist: {fixtures}")
    paths = sorted(
        path
        for path in fixtures.glob("*.jsonl")
        if path.is_file() and path.stat().st_size <= settings.max_upload_bytes
    )
    if not paths:
        raise FileNotFoundError(f"no bounded JSONL trace fixtures were found in {fixtures}")

    jobs: list[dict[str, object]] = []
    for fixture in paths:
        content = fixture.read_bytes()
        source_sha256 = hashlib.sha256(content).hexdigest()

        def persist() -> list[dict[str, object]]:
            ingestion = ingest_jsonl(fixture, source_name=fixture.name)
            records: list[dict[str, object]] = []
            for trace in ingestion.traces:
                evaluation = heuristic_evaluate_trace(trace)
                created = repository.create_job(
                    trace=trace,
                    evaluation=evaluation,
                    source_name=fixture.name,
                    source_sha256=source_sha256,
                    request_sha256=source_sha256,
                    idempotency_key=(
                        f"demo-{source_sha256[:48]}-{trace.trace_id[:24]}"
                    ),
                )
                records.append(
                    {
                        "fixture": fixture.name,
                        "job_id": created.job.job_id,
                        "trace_id": created.job.trace_id,
                        "trace_sha256": created.job.trace_sha256,
                        "evaluation_id": created.job.evaluation_id,
                        "policy_action": str(
                            created.job.evaluation.policy_action.value
                            if hasattr(created.job.evaluation.policy_action, "value")
                            else created.job.evaluation.policy_action
                        ),
                        "created": created.created,
                    }
                )
            return records

        jobs.extend(await asyncio.to_thread(persist))
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = Settings.from_env()
    if arguments.command == "serve":
        settings = Settings(
            database_path=arguments.db,
            artifact_dir=arguments.artifacts,
            standards_path=settings.standards_path,
            template_path=settings.template_path,
            max_upload_bytes=settings.max_upload_bytes,
            max_records=settings.max_records,
            sqlite_timeout_seconds=settings.sqlite_timeout_seconds,
            review_public_key=settings.review_public_key,
            audit_signing_key=settings.audit_signing_key,
            audit_key_path=settings.audit_key_path,
            enable_llm=settings.enable_llm,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
            openrouter_timeout_seconds=settings.openrouter_timeout_seconds,
        )
        application = create_app(settings)
        uvicorn.run(
            application,
            host=arguments.host,
            port=arguments.port,
            log_level=arguments.log_level,
            server_header=False,
            date_header=True,
        )
        return 0

    settings = Settings(
        database_path=arguments.db,
        artifact_dir=arguments.artifacts,
        standards_path=settings.standards_path,
        template_path=settings.template_path,
        max_upload_bytes=settings.max_upload_bytes,
        max_records=settings.max_records,
        sqlite_timeout_seconds=settings.sqlite_timeout_seconds,
        review_public_key=settings.review_public_key,
        audit_signing_key=settings.audit_signing_key,
        audit_key_path=settings.audit_key_path,
        enable_llm=False,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_model=settings.openrouter_model,
        openrouter_timeout_seconds=settings.openrouter_timeout_seconds,
    )
    application = create_app(settings)
    jobs = asyncio.run(_ingest_demo_fixtures(application, arguments.fixtures))
    output = {
        "constitutional_rule": CONSTITUTIONAL_RULE,
        "database": str(settings.database_path),
        "jobs": jobs,
        "message": (
            "Fixtures are ingested and await cryptographically verified human review. "
            "No automated compliance approval was issued."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if arguments.no_serve:
        return 0
    print(
        "Starting dashboard. Export remains blocked until a signed actor=human review is recorded.",
        file=sys.stderr,
    )
    uvicorn.run(
        application,
        host=arguments.host,
        port=arguments.port,
        log_level=arguments.log_level,
        server_header=False,
        date_header=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())