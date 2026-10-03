from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import inspect
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from .evaluator import (
    LLMTraceEvaluator,
    OpenRouterJudge,
    TraceEvaluation,
    heuristic_evaluate_trace,
)
from .models import (
    CONSTITUTIONAL_RULE,
    ExhibitPack,
    HumanReviewSignoff,
    Trace,
    canonical_json_bytes,
    compute_trace_sha256,
    sha256_hex,
    verify_human_review_bytes,
)
from .pack_compiler import COMPILER_ID, COMPILER_VERSION, PACK_MEDIA_TYPE, CompilationError, ExhibitPackCompiler
from .policy import (
    AutomatedComplianceStampProhibited,
    ExportNotAuthorized,
    GovernanceRequest,
    InvalidHumanReviewError as InvalidPolicyHumanReviewError,
    PolicyBindingError,
    PolicyError,
    PolicyRequestAction,
)
from .spans import (
    AmbiguousTraceError,
    EmptyTraceError,
    IngestionResult,
    SpanParseError,
    TraceNotFoundError,
    ingest_jsonl,
)
from .standards import load_harmonized_standards

UTC = timezone.utc
API_VERSION = "1.0.0"
DEFAULT_MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_RECORDS = 100_000
MAX_LIST_PAGE_SIZE = 100
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_IDEMPOTENCY_KEY_LENGTH = 128
MAX_SOURCE_NAME_LENGTH = 160
MAX_TRACES_PER_REQUEST = 32

_JOB_ID_RE = re.compile(r"^job-[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY_RE = re.compile(r"^[\x21-\x7e]+$")
_APPROVED_DECISIONS = {
    "accept",
    "accepted",
    "approval",
    "approve",
    "approved",
    "approved-with-conditions",
    "approved-with-observations",
    "accept-with-conditions",
    "sign-off",
    "signed-off",
}
_REJECTED_DECISIONS = {
    "block",
    "blocked",
    "deny",
    "denied",
    "reject",
    "rejected",
}
_RECEIPT_SIGNATURE_CONTEXT = b"exhibit/api-audit-receipt-signature/v1\x00"
_TRACE_LIMITATIONS = (
    "Evidence is limited to the trace snapshot supplied to this service.",
    "A missing event in the supplied snapshot is not inferred and is not proof that the event never occurred.",
    "Article 50 disclosure detection is diagnostic and is not a separate legal conformity determination.",
)

logger = logging.getLogger("exhibit.api")


class ApiProblem(Exception):
    """A safe, structured API failure suitable for regulator-facing clients."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = dict(details or {})


class RepositoryError(RuntimeError):
    """Base class for API evidence persistence failures."""


class RepositoryConflictError(RepositoryError):
    """An immutable job, review, or idempotency record conflicts with existing evidence."""


class RepositoryIntegrityError(RepositoryError):
    """Persisted evidence is missing, malformed, or fails cryptographic verification."""


class ReviewVerificationConfigurationError(RuntimeError):
    """The server cannot bind a review to a configured cryptographic verification key."""


class InvalidReviewSignatureError(ValueError):
    """A human sign-off did not verify against the configured public key."""


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = Path("receipts.db")
    artifact_dir: Path = Path("artifacts")
    standards_path: Path = Path("fixtures/standards/harmonized_standards.json")
    template_path: Path = Path("web/templates")
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_records: int = DEFAULT_MAX_RECORDS
    sqlite_timeout_seconds: float = 10.0
    review_public_key: bytes | None = None
    audit_signing_key: bytes | None = None
    audit_key_path: Path | None = None
    enable_llm: bool = False
    openrouter_api_key: str | None = None
    openrouter_model: str = "stealth/space-bunny-alpha"
    openrouter_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "database_path", Path(self.database_path).expanduser().resolve())
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir).expanduser().resolve())
        object.__setattr__(self, "standards_path", Path(self.standards_path).expanduser().resolve())
        object.__setattr__(self, "template_path", Path(self.template_path).expanduser().resolve())
        if self.audit_key_path is not None:
            object.__setattr__(self, "audit_key_path", Path(self.audit_key_path).expanduser().resolve())

        if not 1024 <= int(self.max_upload_bytes) <= 256 * 1024 * 1024:
            raise ValueError("max_upload_bytes must be between 1 KiB and 256 MiB")
        if not 1 <= int(self.max_records) <= 1_000_000:
            raise ValueError("max_records must be between 1 and 1,000,000")
        if not 0.1 <= float(self.sqlite_timeout_seconds) <= 120.0:
            raise ValueError("sqlite_timeout_seconds must be between 0.1 and 120")
        if self.review_public_key is not None and len(self.review_public_key) < 16:
            raise ValueError("review_public_key must contain a cryptographic public key")
        if self.audit_signing_key is not None and len(self.audit_signing_key) < 32:
            raise ValueError("audit_signing_key must contain at least 32 random bytes")
        if self.enable_llm and not self.openrouter_api_key:
            raise ValueError("EXHIBIT_ENABLE_LLM requires EXHIBIT_OPENROUTER_API_KEY")

    @property
    def review_key_fingerprint(self) -> str | None:
        if self.review_public_key is None:
            return None
        return sha256_hex(self.review_public_key)

    @property
    def db_path(self) -> Path:
        return self.database_path

    @property
    def export_dir(self) -> Path:
        return self.artifact_dir

    @property
    def max_upload_size(self) -> int:
        return self.max_upload_bytes

    @classmethod
    def from_env(cls) -> Settings:
        project_root = Path(__file__).resolve().parents[1]
        return cls(
            database_path=Path(os.getenv("EXHIBIT_DB_PATH", str(Path.cwd() / "receipts.db"))),
            artifact_dir=Path(os.getenv("EXHIBIT_ARTIFACT_DIR", str(Path.cwd() / "artifacts"))),
            standards_path=Path(
                os.getenv(
                    "EXHIBIT_STANDARDS_PATH",
                    str(project_root / "fixtures" / "standards" / "harmonized_standards.json"),
                )
            ),
            template_path=Path(
                os.getenv(
                    "EXHIBIT_TEMPLATE_PATH",
                    str(project_root / "web" / "templates"),
                )
            ),
            max_upload_bytes=int(
                os.getenv("EXHIBIT_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
            ),
            max_records=int(os.getenv("EXHIBIT_MAX_RECORDS", str(DEFAULT_MAX_RECORDS))),
            sqlite_timeout_seconds=float(os.getenv("EXHIBIT_SQLITE_TIMEOUT", "10")),
            review_public_key=_key_from_environment(
                os.getenv("EXHIBIT_REVIEW_PUBLIC_KEY") or os.getenv("EXHIBIT_REVIEW_PUBLIC_KEY_BASE64")
            ),
            audit_signing_key=_key_from_environment(
                os.getenv("EXHIBIT_AUDIT_SIGNING_KEY")
                or os.getenv("EXHIBIT_AUDIT_SIGNING_KEY_BASE64")
            ),
            audit_key_path=(
                Path(os.environ["EXHIBIT_AUDIT_KEY_PATH"])
                if os.getenv("EXHIBIT_AUDIT_KEY_PATH")
                else None
            ),
            enable_llm=_boolean_from_environment("EXHIBIT_ENABLE_LLM", False),
            openrouter_api_key=os.getenv("EXHIBIT_OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("EXHIBIT_OPENROUTER_MODEL", "stealth/space-bunny-alpha"),
            openrouter_timeout_seconds=float(os.getenv("EXHIBIT_OPENROUTER_TIMEOUT", "20")),
        )


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    status: str
    source_name: str
    source_sha256: str
    request_sha256: str
    trace_id: str
    trace_sha256: str
    evaluation_id: str
    trace: Trace
    evaluation: TraceEvaluation
    review: HumanReviewSignoff | None
    review_verified: bool
    review_public_key: bytes | None
    review_key_fingerprint: str | None
    artifact_path: str | None
    artifact_sha256: str | None
    artifact_size: int | None
    created_at: datetime
    updated_at: datetime
    idempotency_key: str | None


@dataclass(frozen=True, slots=True)
class CreateJobResult:
    job: JobRecord
    created: bool


@dataclass(frozen=True, slots=True)
class ReviewResult:
    job: JobRecord
    created: bool


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    sha256: str
    size: int
    created: bool


def _boolean_from_environment(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalised = raw.strip().casefold()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _key_from_environment(value: str | None) -> bytes | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw)
        except ValueError:
            pass
    padded = raw + "=" * (-len(raw) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("configured cryptographic keys must be hexadecimal or base64url") from exc


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().casefold()).strip("-")


def _normalised_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _model_value(model: Any, *names: str, default: Any = None) -> Any:
    if model is None:
        return default
    if isinstance(model, BaseModel):
        dumped = model.model_dump(mode="python", by_alias=True)
    elif isinstance(model, Mapping):
        dumped = model
    else:
        dumped = {
            name: getattr(model, name)
            for name in dir(model)
            if not name.startswith("_") and not callable(getattr(model, name, None))
        }
    for name in names:
        if name in dumped:
            return dumped[name]
    wanted = {_normalised_key(name) for name in names}
    for key, value in dumped.items():
        if _normalised_key(key) in wanted:
            return value
    return default


def _dump_model(value: Any) -> str:
    if isinstance(value, BaseModel):
        return canonical_json_bytes(value).decode("utf-8")
    return canonical_json_bytes(value).decode("utf-8")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_material(
    *,
    sequence: int,
    event_type: str,
    actor_type: str,
    actor_id: str,
    job_id: str,
    occurred_at: datetime,
    payload_sha256: str,
    previous_receipt_hash: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "event_type": event_type,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "job_id": job_id,
        "occurred_at": occurred_at.astimezone(UTC).isoformat(),
        "payload_sha256": payload_sha256,
        "previous_receipt_hash": previous_receipt_hash,
    }


def _sign_receipt(key: bytes, material: Mapping[str, Any]) -> str:
    signature = hmac.new(
        key,
        _RECEIPT_SIGNATURE_CONTEXT + canonical_json_bytes(material),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _load_or_create_audit_key(explicit_key: bytes | None, key_path: Path) -> tuple[bytes, str]:
    if explicit_key is not None:
        key = bytes(explicit_key)
    elif key_path.exists():
        if key_path.is_symlink() or not key_path.is_file():
            raise RepositoryIntegrityError("audit signing key path is not a regular file")
        encoded = key_path.read_text(encoding="ascii").strip()
        try:
            key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except (UnicodeDecodeError, binascii.Error) as exc:
            raise RepositoryIntegrityError("audit signing key file is malformed") from exc
    else:
        key = secrets.token_bytes(32)
        encoded = base64.urlsafe_b64encode(key).rstrip(b"=").decode("ascii")
        key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            encoded = key_path.read_text(encoding="ascii").strip()
            try:
                key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            except (UnicodeDecodeError, binascii.Error) as exc:
                raise RepositoryIntegrityError("concurrently created audit key is malformed") from exc
        else:
            with os.fdopen(descriptor, "w", encoding="ascii") as key_file:
                key_file.write(encoded)
                key_file.flush()
                os.fsync(key_file.fileno())

    if len(key) < 32:
        raise RepositoryIntegrityError("audit signing key must contain at least 32 bytes")
    try:
        mode = key_path.stat().st_mode
    except FileNotFoundError:
        mode = 0
    if key_path.exists() and mode & 0o077:
        raise RepositoryIntegrityError("audit signing key file must not be accessible by group or other users")
    key_id = f"hmac-sha256:{sha256_hex(key)[:16]}"
    return key, key_id


class EvidenceRepository:
    """Bounded SQLite projection plus an append-only, signed receipt chain.

    Full trace, evaluation, and signed human-review evidence is stored in the
    projection. Receipt rows contain evidence digests and are independently
    protected by a per-job SHA-256 chain and HMAC signatures.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        audit_signing_key: bytes | None = None,
        audit_key_path: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._clock = clock or _utc_now
        self._timeout_seconds = timeout_seconds
        resolved_key_path = (
            Path(audit_key_path).expanduser().resolve()
            if audit_key_path is not None
            else self.database_path.with_name(f"{self.database_path.name}.audit-key")
        )
        self._audit_key, self.audit_key_id = _load_or_create_audit_key(
            audit_signing_key,
            resolved_key_path,
        )
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self._timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = ?", (int(self._timeout_seconds * 1000),))
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA trusted_schema = OFF")
        return connection

    def _initialise(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 1:
                raise RepositoryIntegrityError(
                    f"receipt database schema {version} is newer than supported schema 1"
                )
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS exhibit_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    trace_sha256 TEXT NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    review_json TEXT,
                    review_verified INTEGER NOT NULL DEFAULT 0 CHECK (review_verified IN (0, 1)),
                    review_public_key BLOB,
                    review_key_fingerprint TEXT,
                    artifact_path TEXT,
                    artifact_sha256 TEXT,
                    artifact_size INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE
                );

                CREATE INDEX IF NOT EXISTS exhibit_jobs_trace_id_idx
                    ON exhibit_jobs(trace_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS exhibit_jobs_created_idx
                    ON exhibit_jobs(created_at DESC);

                CREATE TABLE IF NOT EXISTS exhibit_audit_receipts (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                    event_type TEXT NOT NULL,
                    actor_type TEXT NOT NULL CHECK (actor_type IN ('service', 'evaluator', 'human')),
                    actor_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    previous_receipt_hash TEXT,
                    receipt_hash TEXT NOT NULL UNIQUE,
                    signature TEXT NOT NULL,
                    signing_key_id TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES exhibit_jobs(job_id) ON DELETE RESTRICT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS exhibit_receipts_job_sequence_idx
                    ON exhibit_audit_receipts(job_id, sequence);

                CREATE TRIGGER IF NOT EXISTS exhibit_receipts_prevent_update
                BEFORE UPDATE ON exhibit_audit_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'audit receipts are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS exhibit_receipts_prevent_delete
                BEFORE DELETE ON exhibit_audit_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'audit receipts are immutable');
                END;
                """
            )
            connection.execute("PRAGMA user_version = 1")
        except sqlite3.DatabaseError as exc:
            raise RepositoryIntegrityError("receipt database initialisation failed") from exc
        finally:
            connection.close()

        try:
            os.chmod(self.database_path, 0o600)
        except OSError as exc:
            raise RepositoryIntegrityError("receipt database permissions could not be secured") from exc

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise RepositoryIntegrityError("repository clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        try:
            trace = Trace.model_validate_json(row["trace_json"])
            evaluation = TraceEvaluation.model_validate_json(row["evaluation_json"])
            review = (
                HumanReviewSignoff.model_validate_json(row["review_json"])
                if row["review_json"] is not None
                else None
            )
            review_key = bytes(row["review_public_key"]) if row["review_public_key"] else None
            return JobRecord(
                job_id=str(row["job_id"]),
                status=str(row["status"]),
                source_name=str(row["source_name"]),
                source_sha256=str(row["source_sha256"]),
                request_sha256=str(row["request_sha256"]),
                trace_id=str(row["trace_id"]),
                trace_sha256=str(row["trace_sha256"]),
                evaluation_id=str(row["evaluation_id"]),
                trace=trace,
                evaluation=evaluation,
                review=review,
                review_verified=bool(row["review_verified"]),
                review_public_key=review_key,
                review_key_fingerprint=(
                    str(row["review_key_fingerprint"])
                    if row["review_key_fingerprint"] is not None
                    else None
                ),
                artifact_path=(
                    str(row["artifact_path"]) if row["artifact_path"] is not None else None
                ),
                artifact_sha256=(
                    str(row["artifact_sha256"])
                    if row["artifact_sha256"] is not None
                    else None
                ),
                artifact_size=(
                    int(row["artifact_size"])
                    if row["artifact_size"] is not None
                    else None
                ),
                created_at=_parse_datetime(str(row["created_at"])),
                updated_at=_parse_datetime(str(row["updated_at"])),
                idempotency_key=(
                    str(row["idempotency_key"])
                    if row["idempotency_key"] is not None
                    else None
                ),
            )
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            raise RepositoryIntegrityError(
                f"persisted evidence for job '{row['job_id']}' is malformed"
            ) from exc

    def _insert_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        actor_type: str,
        actor_id: str,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        previous = connection.execute(
            """
            SELECT sequence, receipt_hash
            FROM exhibit_audit_receipts
            WHERE job_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        sequence = int(previous["sequence"]) + 1 if previous is not None else 1
        previous_hash = str(previous["receipt_hash"]) if previous is not None else None
        occurred_at = self._now()
        payload_bytes = canonical_json_bytes(payload)
        payload_hash = sha256_hex(payload_bytes)
        material = _event_material(
            sequence=sequence,
            event_type=event_type,
            actor_type=actor_type,
            actor_id=actor_id,
            job_id=job_id,
            occurred_at=occurred_at,
            payload_sha256=payload_hash,
            previous_receipt_hash=previous_hash,
        )
        receipt_hash = sha256_hex(material)
        signature = _sign_receipt(self._audit_key, material)
        connection.execute(
            """
            INSERT INTO exhibit_audit_receipts (
                sequence, schema_version, event_type, actor_type, actor_id,
                job_id, occurred_at, payload_json, payload_sha256,
                previous_receipt_hash, receipt_hash, signature, signing_key_id
            ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                event_type,
                actor_type,
                actor_id,
                job_id,
                occurred_at.isoformat(),
                payload_bytes.decode("utf-8"),
                payload_hash,
                previous_hash,
                receipt_hash,
                signature,
                self.audit_key_id,
            ),
        )
        return receipt_hash

    def get_job(self, job_id: str) -> JobRecord | None:
        if not _JOB_ID_RE.fullmatch(str(job_id)):
            return None
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM exhibit_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return self._row_to_job(row) if row is not None else None
        except sqlite3.DatabaseError as exc:
            raise RepositoryIntegrityError("failed to read persisted job") from exc
        finally:
            connection.close()

    def list_jobs(
        self,
        *,
        job_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        limit = max(1, min(int(limit), MAX_LIST_PAGE_SIZE))
        offset = max(0, int(offset))
        clauses: list[str] = []
        arguments: list[Any] = []
        if job_id:
            clauses.append("job_id = ?")
            arguments.append(job_id)
        if trace_id:
            clauses.append("trace_id = ?")
            arguments.append(trace_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM exhibit_jobs{where}",
                    tuple(arguments),
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT *
                FROM exhibit_jobs{where}
                ORDER BY created_at DESC, job_id DESC
                LIMIT ? OFFSET ?
                """,
                (*arguments, limit, offset),
            ).fetchall()
            return [self._row_to_job(row) for row in rows], total
        except sqlite3.DatabaseError as exc:
            raise RepositoryIntegrityError("failed to list persisted jobs") from exc
        finally:
            connection.close()

    def create_job(
        self,
        *,
        trace: Trace,
        evaluation: TraceEvaluation,
        source_name: str,
        source_sha256: str,
        request_sha256: str,
        idempotency_key: str | None = None,
    ) -> CreateJobResult:
        if not _JOB_ID_RE.fullmatch(str(idempotency_key or "")) and idempotency_key is not None:
            pass
        if idempotency_key is not None and (
            not idempotency_key
            or len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LENGTH
            or not _IDEMPOTENCY_RE.fullmatch(idempotency_key)
        ):
            raise RepositoryConflictError("invalid idempotency key")
        if not _SHA256_RE.fullmatch(source_sha256):
            raise RepositoryConflictError("invalid source SHA-256")
        if not _SHA256_RE.fullmatch(request_sha256):
            raise RepositoryConflictError("invalid request SHA-256")
        if not source_name or len(source_name) > MAX_SOURCE_NAME_LENGTH:
            raise RepositoryConflictError("invalid source name")
        if any(ord(character) < 32 or ord(character) == 127 for character in source_name):
            raise RepositoryConflictError("source name contains control characters")

        computed_trace_hash = compute_trace_sha256(trace)
        if trace.trace_sha256 != computed_trace_hash:
            raise RepositoryConflictError("trace integrity hash does not match trace content")
        if evaluation.trace_sha256 != computed_trace_hash:
            raise RepositoryConflictError("evaluation is not bound to the supplied trace")
        if evaluation.trace_id != trace.trace_id:
            raise RepositoryConflictError("evaluation trace identifier does not match trace")

        if idempotency_key is not None:
            connection = self._connect()
            try:
                existing = connection.execute(
                    "SELECT * FROM exhibit_jobs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            finally:
                connection.close()
            if existing is not None:
                if str(existing["request_sha256"]) != request_sha256:
                    raise RepositoryConflictError(
                        "idempotency key was already used for a different request"
                    )
                return CreateJobResult(job=self._row_to_job(existing), created=False)

        job_id = f"job-{uuid.uuid4().hex}"
        now = self._now()
        trace_json = _dump_model(trace)
        evaluation_json = _dump_model(evaluation)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO exhibit_jobs (
                    job_id, status, source_name, source_sha256, request_sha256,
                    trace_id, trace_sha256, evaluation_id, trace_json,
                    evaluation_json, created_at, updated_at, idempotency_key
                ) VALUES (?, 'awaiting-human-review', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    source_name,
                    source_sha256,
                    request_sha256,
                    trace.trace_id,
                    computed_trace_hash,
                    evaluation.evaluation_id,
                    trace_json,
                    evaluation_json,
                    now.isoformat(),
                    now.isoformat(),
                    idempotency_key,
                ),
            )
            self._insert_receipt(
                connection,
                event_type="trace.ingested",
                actor_type="evaluator",
                actor_id=str(evaluation.engine)[:128],
                job_id=job_id,
                payload={
                    "source_name": source_name,
                    "source_sha256": source_sha256,
                    "request_sha256": request_sha256,
                    "trace_id": trace.trace_id,
                    "trace_sha256": computed_trace_hash,
                    "evaluation_id": evaluation.evaluation_id,
                    "evaluation_trace_sha256": evaluation.trace_sha256,
                    "span_count": len(trace.spans),
                    "injection_detected": evaluation.injection_detected,
                    "injection_caught": evaluation.injection_caught,
                    "policy_action": _model_value(
                        evaluation,
                        "policy_action",
                        default="unknown",
                    ),
                },
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM exhibit_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return CreateJobResult(job=self._row_to_job(row), created=True)
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM exhibit_jobs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None and str(existing["request_sha256"]) == request_sha256:
                    return CreateJobResult(job=self._row_to_job(existing), created=False)
            raise RepositoryConflictError("immutable job identity already exists") from exc
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise RepositoryIntegrityError("failed to persist ingested evidence") from exc
        finally:
            connection.close()

    def record_review(
        self,
        *,
        job_id: str,
        review: HumanReviewSignoff,
        public_key: bytes,
    ) -> ReviewResult:
        if not public_key:
            raise RepositoryConflictError("a reviewer verification key is required")
        if _model_value(review, "signature", "review_signature", "signature_value") is None:
            raise RepositoryConflictError("human review signature is required")

        expected_bindings = {
            "job_id": job_id,
            "trace_id": ("trace_id", "evidence_trace_id"),
            "trace_sha256": ("trace_sha256", "evidence_trace_sha256", "review_target_sha256"),
            "evaluation_id": ("evaluation_id", "evidence_evaluation_id"),
            "evaluation_trace_sha256": (
                "evaluation_trace_sha256",
                "evidence_evaluation_trace_sha256",
            ),
        }
        for expected_name, aliases in expected_bindings.items():
            expected = job_id if expected_name == "job_id" else expected_name
            actual = _model_value(review, *aliases)
            if aliases == ("trace_id", "evidence_trace_id"):
                expected = self.get_job(job_id).trace_id if self.get_job(job_id) else expected
            if actual is None:
                raise RepositoryConflictError(f"human review is missing {expected_name}")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM exhibit_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RepositoryConflictError("job does not exist")
            job = self._row_to_job(row)

            bindings = {
                "job_id": job.job_id,
                "trace_id": job.trace_id,
                "trace_sha256": job.trace_sha256,
                "evaluation_id": job.evaluation_id,
                "evaluation_trace_sha256": job.evaluation_id,
            }
            for field, expected in bindings.items():
                if field == "evaluation_trace_sha256":
                    actual = _model_value(
                        review,
                        "evaluation_trace_sha256",
                        "evidence_evaluation_trace_sha256",
                    )
                else:
                    actual = _model_value(review, field)
                if str(actual) != str(expected):
                    raise RepositoryConflictError(
                        f"human review {field} does not match persisted evidence"
                    )

            review_json = _dump_model(review)
            if row["review_json"] is not None:
                if str(row["review_json"]) == review_json and bytes(row["review_public_key"] or b"") == public_key:
                    return ReviewResult(job=job, created=False)
                raise RepositoryConflictError("human review is immutable and may not be replaced")

            reviewer_id = _model_value(
                review,
                "reviewer_id",
                "actor_id",
                "human_reviewer_id",
                "reviewer_name",
            )
            if not reviewer_id:
                raise RepositoryConflictError("human reviewer identity is required")
            reviewer_id = str(reviewer_id)[:128]
            decision = _normalise_token(
                _model_value(review, "decision", "review_decision", default="")
            )
            status = "ready-for-export" if decision in _APPROVED_DECISIONS else "review-rejected"
            key_fingerprint = sha256_hex(public_key)
            now = self._now()
            connection.execute(
                """
                UPDATE exhibit_jobs
                SET status = ?, review_json = ?, review_verified = 1,
                    review_public_key = ?, review_key_fingerprint = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    review_json,
                    sqlite3.Binary(public_key),
                    key_fingerprint,
                    now.isoformat(),
                    job_id,
                ),
            )
            self._insert_receipt(
                connection,
                event_type="human.review.signed",
                actor_type="human",
                actor_id=reviewer_id,
                job_id=job_id,
                payload={
                    "review_sha256": sha256_hex(review_json),
                    "decision": decision,
                    "trace_id": job.trace_id,
                    "trace_sha256": job.trace_sha256,
                    "evaluation_id": job.evaluation_id,
                    "review_key_fingerprint": key_fingerprint,
                    "constitutional_rule": CONSTITUTIONAL_RULE,
                },
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM exhibit_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return ReviewResult(job=self._row_to_job(updated), created=True)
        except RepositoryConflictError:
            connection.rollback()
            raise
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise RepositoryIntegrityError("failed to persist human review") from exc
        finally:
            connection.close()

    def mark_exported(
        self,
        *,
        job_id: str,
        artifact_path: Path,
        artifact_sha256: str,
        artifact_size: int,
        actor_id: str = COMPILER_ID,
    ) -> JobRecord:
        if not _SHA256_RE.fullmatch(artifact_sha256):
            raise RepositoryConflictError("invalid artifact SHA-256")
        if artifact_size < 1 or artifact_size > MAX_ARTIFACT_BYTES:
            raise RepositoryConflictError("invalid artifact size")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM exhibit_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RepositoryConflictError("job does not exist")
            if row["artifact_sha256"] is not None:
                if (
                    str(row["artifact_sha256"]) != artifact_sha256
                    or str(row["artifact_path"]) != str(artifact_path)
                    or int(row["artifact_size"]) != artifact_size
                ):
                    raise RepositoryConflictError("immutable artifact record conflicts with export")
                return self._row_to_job(row)
            if row["review_json"] is None or not bool(row["review_verified"]):
                raise RepositoryConflictError("verified human review is required before export")
            now = self._now()
            connection.execute(
                """
                UPDATE exhibit_jobs
                SET status = 'exported', artifact_path = ?, artifact_sha256 = ?,
                    artifact_size = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    str(artifact_path),
                    artifact_sha256,
                    artifact_size,
                    now.isoformat(),
                    job_id,
                ),
            )
            self._insert_receipt(
                connection,
                event_type="exhibit.exported",
                actor_type="service",
                actor_id=str(actor_id)[:128],
                job_id=job_id,
                payload={
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": artifact_sha256,
                    "artifact_size": artifact_size,
                    "trace_sha256": str(row["trace_sha256"]),
                    "evaluation_id": str(row["evaluation_id"]),
                    "constitutional_rule": CONSTITUTIONAL_RULE,
                },
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM exhibit_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return self._row_to_job(updated)
        except RepositoryConflictError:
            connection.rollback()
            raise
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise RepositoryIntegrityError("failed to persist exported artifact") from exc
        finally:
            connection.close()

    def verify_audit_chain(self, job_id: str | None = None) -> bool:
        connection = self._connect()
        try:
            if job_id is None:
                rows = connection.execute(
                    "SELECT * FROM exhibit_audit_receipts ORDER BY job_id, sequence"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM exhibit_audit_receipts
                    WHERE job_id = ?
                    ORDER BY sequence
                    """,
                    (job_id,),
                ).fetchall()
            expected_sequence: dict[str, int] = {}
            expected_previous: dict[str, str | None] = {}
            for row in rows:
                current_job = str(row["job_id"])
                sequence = expected_sequence.get(current_job, 0) + 1
                if int(row["sequence"]) != sequence:
                    return False
                expected_sequence[current_job] = sequence
                previous = expected_previous.get(current_job)
                if row["previous_receipt_hash"] != previous:
                    return False
                payload = str(row["payload_json"]).encode("utf-8")
                payload_hash = sha256_hex(payload)
                if payload_hash != str(row["payload_sha256"]):
                    return False
                material = _event_material(
                    sequence=sequence,
                    event_type=str(row["event_type"]),
                    actor_type=str(row["actor_type"]),
                    actor_id=str(row["actor_id"]),
                    job_id=current_job,
                    occurred_at=_parse_datetime(str(row["occurred_at"])),
                    payload_sha256=payload_hash,
                    previous_receipt_hash=previous,
                )
                receipt_hash = sha256_hex(material)
                if receipt_hash != str(row["receipt_hash"]):
                    return False
                expected_signature = _sign_receipt(self._audit_key, material)
                if not hmac.compare_digest(expected_signature, str(row["signature"])):
                    return False
                expected_previous[current_job] = receipt_hash
            return True
        except (sqlite3.DatabaseError, RepositoryIntegrityError, ValueError, TypeError):
            return False
        finally:
            connection.close()

    def health(self) -> bool:
        connection = self._connect()
        try:
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            required = {
                row["name"]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('exhibit_jobs', 'exhibit_audit_receipts')
                    """
                ).fetchall()
            }
            return quick_check == "ok" and required == {
                "exhibit_jobs",
                "exhibit_audit_receipts",
            }
        except sqlite3.DatabaseError:
            return False
        finally:
            connection.close()


class ReviewVerificationUnavailable(Exception):
    """A verifier signature could not be bound to supported parameters."""


def _verifier_call(
    verifier: Callable[..., Any],
    *,
    review: Any,
    payload: bytes,
    signature: str,
    public_key: bytes,
) -> Any:
    try:
        parameters = inspect.signature(verifier).parameters
    except (TypeError, ValueError) as exc:
        raise ReviewVerificationUnavailable("review verifier has no inspectable signature") from exc

    values: dict[str, Any] = {
        "review": review,
        "signoff": review,
        "payload": payload,
        "message": payload,
        "data": payload,
        "bytes": payload,
        "signedbytes": payload,
        "payloadbytes": payload,
        "canonicalpayload": payload,
        "signature": signature,
        "encodedsignature": signature,
        "signaturevalue": signature,
        "publickey": public_key,
        "verificationkey": public_key,
        "signingpublickey": public_key,
        "key": public_key,
    }
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    has_var_keyword = False
    usable = False

    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend((review, payload, signature, public_key))
            usable = True
            continue
        normalised = _normalised_key(name)
        value = values.get(normalised)
        if value is None and normalised in {"public", "verifierkey"}:
            value = public_key
        if value is None and normalised in {"signed", "signingpayload", "content"}:
            value = payload
        if value is None and normalised in {"humanreview", "reviewsignoff"}:
            value = review
        if value is not None:
            usable = True
        elif parameter.default is inspect.Parameter.empty:
            raise ReviewVerificationUnavailable(
                f"review verifier parameter '{name}' is unsupported"
            )
        else:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[name] = value

    if not usable:
        if has_var_keyword:
            keywords.update(
                {
                    "payload": payload,
                    "signature": signature,
                    "public_key": public_key,
                }
            )
        else:
            raise ReviewVerificationUnavailable("review verifier accepts no supported evidence")
    return verifier(*positional, **keywords)


def verify_signed_human_review(review: Any, public_key: bytes) -> None:
    """Verify a signed review without ever creating or modifying its signature."""

    if not public_key:
        raise ReviewVerificationConfigurationError("no reviewer public key is configured")
    signature_value = _model_value(
        review,
        "signature",
        "review_signature",
        "signature_value",
    )
    if not signature_value or not isinstance(signature_value, str):
        raise InvalidReviewSignatureError("human review does not contain a textual signature")
    signature = signature_value

    payloads: list[bytes] = []
    for attribute in (
        "signing_payload",
        "signed_payload",
        "signing_bytes",
        "signed_bytes",
        "payload_bytes",
        "canonical_payload",
    ):
        try:
            value = getattr(review, attribute)
            if callable(value):
                value = value()
            if isinstance(value, bytes):
                payloads.append(value)
            elif isinstance(value, str):
                payloads.append(value.encode("utf-8"))
            elif isinstance(value, Mapping):
                payloads.append(canonical_json_bytes(value))
        except (AttributeError, TypeError, ValueError):
            continue

    dumped = review.model_dump(mode="json", by_alias=True)
    signature_keys = {
        key
        for key in dumped
        if _normalised_key(key) in {"signature", "reviewsignature", "signaturevalue"}
    }
    public_keys = {
        key
        for key in dumped
        if _normalised_key(key)
        in {"publickey", "verificationkey", "signingpublickey", "key"}
    }
    metadata_keys = {
        key
        for key in dumped
        if _normalised_key(key)
        in {"signaturealgorithm", "publickeyfingerprint", "reviewkeyfingerprint"}
    }
    payloads.extend(
        (
            canonical_json_bytes(
                {key: value for key, value in dumped.items() if key not in signature_keys}
            ),
            canonical_json_bytes(
                {
                    key: value
                    for key, value in dumped.items()
                    if key not in signature_keys | public_keys | metadata_keys
                }
            ),
        )
    )
    unique_payloads: list[bytes] = []
    seen: set[bytes] = set()
    for payload in payloads:
        if payload not in seen:
            seen.add(payload)
            unique_payloads.append(payload)

    verifiers: list[Callable[..., Any]] = []
    for method_name in ("verify_signature", "verify"):
        method = getattr(review, method_name, None)
        if callable(method):
            verifiers.append(method)
    verifiers.append(verify_human_review_bytes)

    invoked = False
    for verifier in verifiers:
        for payload in unique_payloads:
            try:
                result = _verifier_call(
                    verifier,
                    review=review,
                    payload=payload,
                    signature=signature,
                    public_key=public_key,
                )
            except ReviewVerificationUnavailable:
                continue
            except Exception:
                continue
            invoked = True
            if isinstance(result, tuple):
                result = result[0] if result else False
            if hasattr(result, "valid"):
                result = result.valid
            if result is True:
                return
    if not invoked:
        raise ReviewVerificationConfigurationError(
            "the configured human-review verifier cannot be called safely"
        )
    raise InvalidReviewSignatureError(
        "human review signature verification failed against the configured public key"
    )


def _validate_review_binding(
    review: Any,
    job: JobRecord,
) -> tuple[str, str]:
    actor_type = _model_value(review, "actor_type", "reviewer_type", "subject_type")
    if actor_type is not None and _normalise_token(actor_type) != "human":
        raise ApiProblem(
            422,
            "actor_must_be_human",
            "Human-review sign-offs must identify the reviewer as actor type human.",
        )

    bindings = {
        "job_id": job.job_id,
        "trace_id": job.trace_id,
        "trace_sha256": job.trace_sha256,
        "evaluation_id": job.evaluation_id,
    }
    aliases = {
        "trace_sha256": (
            "trace_sha256",
            "evidence_trace_sha256",
            "review_target_sha256",
        ),
        "evaluation_trace_sha256": (
            "evaluation_trace_sha256",
            "evidence_evaluation_trace_sha256",
        ),
    }
    for field, expected in bindings.items():
        actual = _model_value(review, field)
        if actual is None or str(actual) != str(expected):
            raise ApiProblem(
                422,
                "review_binding_mismatch",
                f"Human review {field} does not match the persisted evidence.",
                details={"field": field, "expected": expected},
            )
    actual_evaluation_hash = _model_value(review, *aliases["evaluation_trace_sha256"])
    if actual_evaluation_hash is None or str(actual_evaluation_hash) != job.trace_sha256:
        raise ApiProblem(
            422,
            "review_binding_mismatch",
            "Human review evaluation_trace_sha256 does not match the persisted trace.",
            details={"field": "evaluation_trace_sha256", "expected": job.trace_sha256},
        )

    reviewer_id = _model_value(
        review,
        "reviewer_id",
        "actor_id",
        "human_reviewer_id",
        "reviewer_name",
    )
    if not reviewer_id:
        raise ApiProblem(
            422,
            "reviewer_identity_required",
            "A human reviewer identity is required.",
        )
    review_id = _model_value(review, "review_id", "signoff_id")
    if not review_id:
        raise ApiProblem(
            422,
            "review_id_required",
            "A stable human review identifier is required.",
        )
    decision = _normalise_token(
        _model_value(review, "decision", "review_decision", default="")
    )
    if not decision:
        raise ApiProblem(
            422,
            "review_decision_required",
            "A human review decision is required.",
        )
    if decision not in _APPROVED_DECISIONS | _REJECTED_DECISIONS:
        raise ApiProblem(
            422,
            "unsupported_review_decision",
            "The review decision is not recognised by the governance policy.",
        )
    return str(reviewer_id), decision


def _construct_default_compiler(settings: Settings) -> Any:
    catalog = load_harmonized_standards(settings.standards_path)
    constructor = ExhibitPackCompiler
    try:
        parameters = inspect.signature(constructor).parameters
    except (TypeError, ValueError):
        return constructor(catalog)

    for name in ("catalog", "standards_catalog", "catalogue"):
        if name in parameters:
            return constructor(**{name: catalog})
    for name in ("standards_path", "catalog_path", "catalogue_path", "path"):
        if name in parameters:
            return constructor(**{name: settings.standards_path})
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return constructor(catalog=catalog)
    return constructor()


def _invoke_compiler(
    compiler: Any,
    *,
    trace: Trace,
    evaluation: TraceEvaluation,
    review: HumanReviewSignoff,
    job_id: str,
    public_key: bytes,
    catalog: Any,
) -> Any:
    method = getattr(compiler, "compile", None)
    if not callable(method):
        method = getattr(compiler, "compile_pack", None)
    if not callable(method):
        raise CompilationError("configured compiler does not expose compile or compile_pack")

    request = GovernanceRequest(
        trace=trace,
        evaluation=evaluation,
        human_review=review,
        job_id=job_id,
        public_key=public_key,
        requested_action=PolicyRequestAction.EXPORT,
    )
    standard_values: dict[str, Any] = {
        "request": request,
        "governance_request": request,
        "trace": trace,
        "evaluation": evaluation,
        "evaluation_result": evaluation,
        "human_review": review,
        "review": review,
        "signoff": review,
        "job_id": job_id,
        "public_key": public_key,
        "verification_key": public_key,
        "signing_public_key": public_key,
        "catalog": catalog,
        "standards_catalog": catalog,
        "requested_action": PolicyRequestAction.EXPORT,
        "policy_action": PolicyRequestAction.EXPORT,
    }
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError) as exc:
        raise CompilationError("configured compiler has an unusable method signature") from exc

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    recognised = 0
    has_varargs = False
    has_var_keyword = False
    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        value = standard_values.get(_normalised_key(name))
        if value is None:
            value = standard_values.get(name)
        if value is not None:
            recognised += 1
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[name] = value
        elif parameter.default is inspect.Parameter.empty:
            raise CompilationError(
                f"configured compiler requires unsupported parameter '{name}'"
            )

    if has_varargs and recognised == 0:
        args.extend((trace, evaluation, review, job_id, public_key))
    elif has_var_keyword and recognised == 0:
        kwargs.update(
            {
                "trace": trace,
                "evaluation": evaluation,
                "human_review": review,
                "job_id": job_id,
                "public_key": public_key,
            }
        )
    return method(*args, **kwargs)


def _pack_from_compiler_result(result: Any) -> ExhibitPack:
    if isinstance(result, ExhibitPack):
        return result
    if isinstance(result, bytes):
        return ExhibitPack.model_validate_json(result)
    if isinstance(result, str):
        return ExhibitPack.model_validate_json(result)
    if isinstance(result, Mapping):
        return ExhibitPack.model_validate(result)
    for attribute in ("pack", "exhibit_pack", "compiled_pack", "evidence_pack"):
        candidate = getattr(result, attribute, None)
        if callable(candidate):
            candidate = candidate()
        if isinstance(candidate, ExhibitPack):
            return candidate
        if isinstance(candidate, Mapping):
            return ExhibitPack.model_validate(candidate)
    artifact_path = getattr(result, "artifact_path", None)
    if artifact_path is not None:
        path = Path(artifact_path)
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_ARTIFACT_BYTES:
            return ExhibitPack.model_validate_json(path.read_bytes())
    raise CompilationError("compiler result does not contain a valid ExhibitPack")


class ArtifactService:
    def __init__(
        self,
        settings: Settings,
        repository: EvidenceRepository,
        *,
        compiler: Any | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.catalog = load_harmonized_standards(settings.standards_path)
        self.compiler = compiler or _construct_default_compiler(settings)

    def _path_for(self, job_id: str) -> Path:
        return self.settings.artifact_dir / job_id / "exhibit.json"

    def _read_verified(self, job: JobRecord) -> tuple[Path, bytes, str]:
        if not job.artifact_path or not job.artifact_sha256:
            raise RepositoryIntegrityError("job has no recorded immutable exhibit artifact")
        path = Path(job.artifact_path)
        expected_path = self._path_for(job.job_id)
        if path != expected_path:
            raise RepositoryIntegrityError("recorded exhibit path is outside its immutable job directory")
        if path.is_symlink() or not path.is_file():
            raise RepositoryIntegrityError("recorded exhibit artifact is missing or is not a regular file")
        size = path.stat().st_size
        if size < 1 or size > MAX_ARTIFACT_BYTES:
            raise RepositoryIntegrityError("recorded exhibit artifact has an invalid size")
        content = path.read_bytes()
        digest = sha256_hex(content)
        if digest != job.artifact_sha256:
            raise RepositoryIntegrityError("recorded exhibit artifact failed SHA-256 verification")
        return path, content, digest

    def _write_immutable(self, job_id: str, content: bytes) -> Path:
        final_path = self._path_for(job_id)
        final_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            directory_stat = final_path.parent.lstat()
        except OSError as exc:
            raise RepositoryIntegrityError("artifact job directory cannot be inspected") from exc
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            raise RepositoryIntegrityError("artifact job path is not a regular directory")

        if final_path.exists() or final_path.is_symlink():
            if final_path.is_symlink() or not final_path.is_file():
                raise RepositoryIntegrityError("existing exhibit path is not a regular file")
            existing = final_path.read_bytes()
            if existing != content:
                raise RepositoryIntegrityError(
                    "an immutable exhibit already exists with different content"
                )
            return final_path

        temporary_path = final_path.with_name(f".exhibit-{secrets.token_hex(12)}.tmp")
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as artifact:
                artifact.write(content)
                artifact.flush()
                os.fsync(artifact.fileno())
            try:
                os.link(temporary_path, final_path, follow_symlinks=False)
            except FileExistsError:
                existing = final_path.read_bytes() if final_path.is_file() else b""
                if existing != content:
                    raise RepositoryIntegrityError(
                        "concurrent immutable exhibit creation produced different content"
                    )
            except OSError:
                descriptor = os.open(
                    final_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as artifact:
                    artifact.write(content)
                    artifact.flush()
                    os.fsync(artifact.fileno())
            directory_descriptor = os.open(final_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            return final_path
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def export_job(self, job: JobRecord) -> ExportResult:
        if job.review is None or not job.review_verified:
            raise RepositoryConflictError("verified human review is required before export")
        if not job.review_public_key:
            raise RepositoryConflictError("review verification key is unavailable")
        if job.status == "exported":
            path, content, digest = self._read_verified(job)
            return ExportResult(path, digest, len(content), created=False)

        result = _invoke_compiler(
            self.compiler,
            trace=job.trace,
            evaluation=job.evaluation,
            review=job.review,
            job_id=job.job_id,
            public_key=job.review_public_key,
            catalog=self.catalog,
        )
        pack = _pack_from_compiler_result(result)
        if _model_value(pack, "constitutional_rule") != CONSTITUTIONAL_RULE:
            raise CompilationError("compiler output omitted the constitutional limitation")
        content = canonical_json_bytes(pack)
        if len(content) > MAX_ARTIFACT_BYTES:
            raise CompilationError("compiled exhibit exceeds the artifact size boundary")
        path = self._write_immutable(job.job_id, content)
        digest = sha256_hex(content)
        self.repository.mark_exported(
            job_id=job.job_id,
            artifact_path=path,
            artifact_sha256=digest,
            artifact_size=len(content),
        )
        return ExportResult(path, digest, len(content), created=True)

    def read_artifact(self, job: JobRecord) -> tuple[Path, bytes, str]:
        return self._read_verified(job)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key '{key}'")
        result[key] = value
    return result


def _strict_json_loads(content: str) -> Any:
    return json.loads(content, object_pairs_hook=_duplicate_rejecting_object)


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().casefold()


def _declared_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ApiProblem(400, "invalid_content_length", "Content-Length must be an integer.") from exc
    if value < 0:
        raise ApiProblem(400, "invalid_content_length", "Content-Length cannot be negative.")
    return value


def _safe_source_name(value: str | None, fallback: str = "upload.jsonl") -> str:
    candidate = Path(str(value or fallback).replace("\\", "/")).name.strip()
    if not candidate or len(candidate) > MAX_SOURCE_NAME_LENGTH:
        raise ApiProblem(422, "invalid_source_name", "The upload source name is invalid.")
    if any(ord(character) < 32 or ord(character) == 127 for character in candidate):
        raise ApiProblem(422, "invalid_source_name", "The upload source name contains control characters.")
    return candidate


async def _read_limited_body(request: Request, maximum: int) -> bytes:
    declared = _declared_length(request)
    if declared is not None and declared > maximum:
        raise ApiProblem(
            413,
            "payload_too_large",
            "The request exceeds the configured evidence upload boundary.",
            details={"maximum_bytes": maximum},
        )
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > maximum:
            raise ApiProblem(
                413,
                "payload_too_large",
                "The request exceeds the configured evidence upload boundary.",
                details={"maximum_bytes": maximum},
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_multipart_upload(request: Request, maximum: int) -> tuple[bytes, str]:
    declared = _declared_length(request)
    if declared is None:
        raise ApiProblem(
            411,
            "content_length_required",
            "A bounded Content-Length is required for multipart evidence uploads.",
        )
    if declared > maximum:
        raise ApiProblem(
            413,
            "payload_too_large",
            "The multipart upload exceeds the configured evidence boundary.",
            details={"maximum_bytes": maximum},
        )
    try:
        try:
            form = await request.form(max_files=2, max_fields=10, max_part_size=maximum)
        except TypeError:
            form = await request.form(max_files=2, max_fields=10)
    except RuntimeError as exc:
        raise ApiProblem(
            415,
            "multipart_support_unavailable",
            "The server's multipart parser is unavailable or rejected the request.",
        ) from exc
    uploads = [item for item in form.values() if isinstance(item, UploadFile)]
    if len(uploads) != 1:
        for item in uploads:
            await item.close()
        raise ApiProblem(
            422,
            "one_trace_file_required",
            "Upload exactly one JSON or JSONL evidence file per request.",
        )
    upload = uploads[0]
    try:
        if upload.size is not None and upload.size > maximum:
            raise ApiProblem(
                413,
                "payload_too_large",
                "The uploaded file exceeds the configured evidence boundary.",
                details={"maximum_bytes": maximum},
            )
        content = await upload.read(maximum + 1)
        if len(content) > maximum:
            raise ApiProblem(
                413,
                "payload_too_large",
                "The uploaded file exceeds the configured evidence boundary.",
                details={"maximum_bytes": maximum},
            )
        return content, _safe_source_name(upload.filename)
    finally:
        await upload.close()


def _prepare_json_source(content: bytes, maximum_records: int) -> tuple[str, int]:
    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ApiProblem(422, "invalid_utf8", "Evidence files must be UTF-8 encoded.") from exc

    try:
        document = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError):
        record_count = sum(1 for line in text.splitlines() if line.strip())
        if record_count > maximum_records:
            raise ApiProblem(
                413,
                "too_many_records",
                "The evidence file exceeds the configured record boundary.",
                details={"maximum_records": maximum_records},
            )
        return text, record_count

    inherited_trace_id: Any = None
    if isinstance(document, Mapping):
        inherited_trace_id = document.get("trace_id") or document.get("traceId")
        for key in ("spans", "records"):
            if key in document:
                records = document[key]
                break
        else:
            if isinstance(document.get("payload"), list):
                records = document["payload"]
            else:
                records = [document]
    elif isinstance(document, list):
        records = document
    else:
        raise ApiProblem(
            422,
            "invalid_json_document",
            "The JSON evidence document must be an object or array of span objects.",
        )

    if not isinstance(records, list) or not records:
        raise ApiProblem(422, "empty_trace", "No executable span evidence was supplied.")
    if len(records) > maximum_records:
        raise ApiProblem(
            413,
            "too_many_records",
            "The evidence document exceeds the configured record boundary.",
            details={"maximum_records": maximum_records},
        )
    prepared: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ApiProblem(
                422,
                "invalid_span_record",
                f"Span record {index + 1} is not a JSON object.",
            )
        enriched = dict(record)
        if inherited_trace_id is not None and not (
            enriched.get("trace_id") or enriched.get("traceId")
        ):
            enriched["trace_id"] = inherited_trace_id
        prepared.append(canonical_json_bytes(enriched).decode("utf-8"))
    return "\n".join(prepared) + "\n", len(prepared)


def _idempotency_for_trace(base_key: str, trace_id: str) -> str:
    suffix = sha256_hex(trace_id.encode("utf-8"))[:20]
    prefix_length = MAX_IDEMPOTENCY_KEY_LENGTH - len(suffix) - 1
    return f"{base_key[:prefix_length]}:{suffix}"


def _evaluate_with_optional_llm(
    trace: Trace,
    settings: Settings,
) -> tuple[TraceEvaluation, list[str]]:
    deterministic = heuristic_evaluate_trace(trace)
    if not settings.enable_llm:
        return deterministic, []
    if not settings.openrouter_api_key:
        return deterministic, [
            "LLM evaluation was requested but no OpenRouter credential is configured; deterministic evaluation was used."
        ]
    try:
        try:
            judge = OpenRouterJudge(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                timeout_seconds=settings.openrouter_timeout_seconds,
            )
        except TypeError:
            judge = OpenRouterJudge(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                timeout=settings.openrouter_timeout_seconds,
            )
        llm_evaluator = LLMTraceEvaluator(judge)
        candidate = llm_evaluator.evaluate(trace)
        if isinstance(candidate, Mapping):
            candidate = TraceEvaluation.model_validate(candidate)
        if not isinstance(candidate, TraceEvaluation):
            raise ValueError("LLM evaluator returned an unsupported result")
        if candidate.trace_sha256 != deterministic.trace_sha256:
            raise ValueError("LLM evaluator result is not bound to the trace")
        return candidate, []
    except Exception:
        logger.warning("LLM evaluation failed; using deterministic evaluator", exc_info=True)
        return deterministic, [
            "The optional LLM evaluator was unavailable or invalid; deterministic offline evaluation was used."
        ]


def _ingest_and_persist(
    *,
    content: bytes,
    source_name: str,
    requested_trace_id: str | None,
    idempotency_key: str | None,
    settings: Settings,
    repository: EvidenceRepository,
) -> tuple[list[CreateJobResult], list[str], IngestionResult]:
    if len(content) > settings.max_upload_bytes:
        raise ApiProblem(413, "payload_too_large", "The evidence file is too large.")
    source_sha256 = hashlib.sha256(content).hexdigest()
    content_type = "application/x-ndjson"
    prepared, _record_count = _prepare_json_source(content, settings.max_records)
    try:
        ingestion = ingest_jsonl(io.StringIO(prepared), source_name=source_name)
    except EmptyTraceError as exc:
        raise ApiProblem(422, "empty_trace", "No executable span evidence was found.") from exc
    except DuplicateSpanError as exc:
        raise ApiProblem(
            422,
            "duplicate_span",
            "The evidence contains a conflicting duplicate span.",
            details={"source": exc.issue.source, "record": exc.issue.record, "reason": exc.issue.reason},
        ) from exc
    except SpanParseError as exc:
        raise ApiProblem(
            422,
            "invalid_span",
            "A supplied span could not be converted into trustworthy evidence.",
            details={
                "source": exc.issue.source,
                "record": exc.issue.record,
                "code": exc.issue.code,
                "reason": exc.issue.reason[:300],
            },
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ApiProblem(422, "invalid_trace_source", str(exc)[:300]) from exc

    if requested_trace_id:
        try:
            selected = (ingestion.get_trace(requested_trace_id),)
        except TraceNotFoundError as exc:
            raise ApiProblem(
                422,
                "trace_not_found",
                "The requested trace identifier is absent from the upload.",
                details={"available_trace_ids": list(exc.available)},
            ) from exc
    else:
        selected = ingestion.traces
    if not selected:
        raise ApiProblem(422, "empty_trace", "No executable trace evidence was found.")
    if len(selected) > MAX_TRACES_PER_REQUEST:
        raise ApiProblem(
            413,
            "too_many_traces",
            f"A single request may contain at most {MAX_TRACES_PER_REQUEST} traces.",
        )

    created: list[CreateJobResult] = []
    warnings: list[str] = []
    for trace in selected:
        evaluation, evaluation_warnings = _evaluate_with_optional_llm(trace, settings)
        key = (
            idempotency_key
            if len(selected) == 1
            else (
                _idempotency_for_trace(idempotency_key, trace.trace_id)
                if idempotency_key is not None
                else None
            )
        )
        created.append(
            repository.create_job(
                trace=trace,
                evaluation=evaluation,
                source_name=source_name,
                source_sha256=source_sha256,
                request_sha256=source_sha256,
                idempotency_key=key,
            )
        )
        warnings.extend(evaluation_warnings)
    return created, list(dict.fromkeys(warnings)), ingestion


def _decision_is_approved(review: Any) -> bool:
    return (
        _normalise_token(_model_value(review, "decision", "review_decision", default=""))
        in _APPROVED_DECISIONS
    )


def _article_readiness(job: JobRecord) -> dict[str, dict[str, Any]]:
    span_count = len(job.trace.spans)
    logging_ready = span_count > 0 and bool(_SHA256_RE.fullmatch(job.trace_sha256))
    approved = job.review_verified and job.review is not None and _decision_is_approved(job.review)
    rejected = job.review is not None and not approved
    threat_flagged = bool(
        _model_value(job.evaluation, "article_15_threat_flagged", default=False)
    )
    policy_action = _normalise_token(
        _model_value(job.evaluation, "policy_action", default="unknown")
    )

    if approved:
        oversight = {
            "status": "ready",
            "score": 100,
            "label": "Human review verified",
            "detail": "A signed actor=human review is bound to this trace and evaluation.",
            "evidence": [
                f"Review key fingerprint: {job.review_key_fingerprint}",
                "Export is gated by deterministic governance policy.",
            ],
        }
    elif rejected:
        oversight = {
            "status": "blocked",
            "score": 0,
            "label": "Human review rejected",
            "detail": "The signed reviewer decision does not authorize export.",
            "evidence": ["A replacement review is prohibited for this immutable job."],
        }
    else:
        oversight = {
            "status": "pending",
            "score": 0,
            "label": "Human review required",
            "detail": "A cryptographically verified actor=human sign-off is required before export.",
            "evidence": ["Automated compliance approval is prohibited."],
        }

    if threat_flagged:
        security = {
            "status": "attention" if policy_action != "action-block" else "blocked",
            "score": 0,
            "label": "Threat signal recorded",
            "detail": "The supplied trace contains a cybersecurity or robustness signal requiring human disposition.",
            "evidence": [
                f"Policy action: {_model_value(job.evaluation, 'policy_action', default='unknown')}",
                f"Threat signals: {len(_model_value(job.evaluation, 'threats', default=()) or ())}",
            ],
        }
    else:
        security = {
            "status": "ready",
            "score": 100,
            "label": "No threat signal in snapshot",
            "detail": "No configured threat signal was observed in the supplied trace snapshot.",
            "evidence": ["Absence in this snapshot is not proof of system-wide robustness."],
        }

    statuses = [
        "ready" if logging_ready else "attention",
        oversight["status"],
        security["status"],
    ]
    if "blocked" in statuses:
        overall = "blocked"
    elif "attention" in statuses:
        overall = "attention"
    elif "pending" in statuses:
        overall = "pending"
    else:
        overall = "ready"
    return {
        "article_12": {
            "status": "ready" if logging_ready else "attention",
            "score": 100 if logging_ready else 0,
            "label": "Trace evidence captured" if logging_ready else "Trace evidence missing",
            "detail": (
                "The supplied execution spans and trace envelope have a SHA-256 evidence digest."
                if logging_ready
                else "No trustworthy executable span evidence is bound to this job."
            ),
            "evidence": [
                f"Span records: {span_count}",
                f"Trace SHA-256: {job.trace_sha256}",
            ],
        },
        "article_14": oversight,
        "article_15": security,
        "overall": {
            "status": overall,
            "label": "Evidence export " + overall,
            "constitutional_rule": CONSTITUTIONAL_RULE,
        },
    }


def _public_job(job: JobRecord, *, include_spans: bool, include_evidence: bool) -> dict[str, Any]:
    trace_summary: dict[str, Any] = {
        "trace_id": job.trace_id,
        "trace_sha256": job.trace_sha256,
        "span_count": len(job.trace.spans),
        "root_span_count": sum(1 for span in job.trace.spans if _model_value(span, "parent_span_id") is None),
        "format": _model_value(job.trace, "format", "trace_format", default="openinference"),
    }
    if include_spans:
        trace_summary["spans"] = [
            span.model_dump(mode="json", by_alias=True) for span in job.trace.spans
        ]

    evaluation_summary: dict[str, Any] = {
        "evaluation_id": job.evaluation_id,
        "engine": _model_value(job.evaluation, "engine"),
        "engine_version": _model_value(job.evaluation, "engine_version"),
        "injection_detected": bool(
            _model_value(job.evaluation, "injection_detected", default=False)
        ),
        "injection_caught": bool(
            _model_value(job.evaluation, "injection_caught", default=False)
        ),
        "schema_valid": bool(_model_value(job.evaluation, "schema_valid", default=False)),
        "disclosure_present": bool(
            _model_value(job.evaluation, "disclosure_present", default=False)
        ),
        "article_15_threat_flagged": bool(
            _model_value(job.evaluation, "article_15_threat_flagged", default=False)
        ),
        "policy_action": _model_value(
            job.evaluation,
            "policy_action",
            default="unknown",
        ),
        "human_review_required": bool(
            _model_value(job.evaluation, "human_review_required", default=True)
        ),
        "overall_score": _model_value(job.evaluation, "overall_score", default=0),
        "confidence": _model_value(job.evaluation, "confidence", default=0),
        "finding_count": len(_model_value(job.evaluation, "findings", default=()) or ()),
        "threat_count": len(_model_value(job.evaluation, "threats", default=()) or ()),
        "llm_status": _model_value(job.evaluation, "llm_status", default="not-requested"),
    }
    if include_evidence:
        evaluation_summary["evidence"] = job.evaluation.model_dump(mode="json", by_alias=True)

    review_summary: dict[str, Any] | None = None
    if job.review is not None:
        signature = str(
            _model_value(job.review, "signature", "review_signature", default="")
        )
        review_summary = {
            "review_id": _model_value(job.review, "review_id", "signoff_id"),
            "reviewer_id": _model_value(
                job.review,
                "reviewer_id",
                "actor_id",
                "human_reviewer_id",
                "reviewer_name",
            ),
            "actor_type": "human",
            "decision": _model_value(job.review, "decision", "review_decision"),
            "reviewed_at": _model_value(job.review, "reviewed_at", "decision_at"),
            "verified": job.review_verified,
            "signature_fingerprint": sha256_hex(signature),
            "review_key_fingerprint": job.review_key_fingerprint,
        }

    readiness = _article_readiness(job)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "source": {
            "name": job.source_name,
            "sha256": job.source_sha256,
        },
        "trace": trace_summary,
        "evaluation": evaluation_summary,
        "review": review_summary,
        "readiness": readiness,
        "policy_action": evaluation_summary["policy_action"],
        "human_review_required_for_export": job.review is None or not approved,
        "download_ready": job.review_verified and approved,
        "download_url": f"/exhibit/{job.job_id}",
        "artifact": (
            {
                "sha256": job.artifact_sha256,
                "size": job.artifact_size,
                "media_type": PACK_MEDIA_TYPE,
            }
            if job.artifact_sha256 is not None
            else None
        ),
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "constitutional_rule": CONSTITUTIONAL_RULE,
        "limitations": list(_TRACE_LIMITATIONS),
    }


def _error_body(request: Request, code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        },
        "constitutional_rule": CONSTITUTIONAL_RULE,
    }
    if details is not None:
        body["error"]["details"] = details
    return body


def _policy_decision_details(error: ExportNotAuthorized) -> dict[str, Any]:
    decision = error.decision
    as_dict = getattr(decision, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    if hasattr(decision, "__dataclass_fields__"):
        return asdict(decision)
    return {
        "authorized": getattr(decision, "authorized", False),
        "violation_codes": list(getattr(decision, "violation_codes", ())),
    }


def create_app(
    settings: Settings | None = None,
    *,
    repository: EvidenceRepository | None = None,
    compiler: Any | None = None,
    database_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> FastAPI:
    """Create a configured FastAPI application without opening network listeners."""

    effective_settings = settings or Settings.from_env()
    if database_path is not None or artifact_dir is not None:
        import dataclasses

        effective_settings = dataclasses.replace(
            effective_settings,
            database_path=Path(database_path) if database_path is not None else effective_settings.database_path,
            artifact_dir=Path(artifact_dir) if artifact_dir is not None else effective_settings.artifact_dir,
        )

    template_directory = effective_settings.template_path
    if not (template_directory / "dashboard.html").is_file():
        raise RuntimeError(f"dashboard template is missing from {template_directory}")
    templates = Jinja2Templates(directory=str(template_directory), autoescape=True)
    evidence_repository = repository or EvidenceRepository(
        effective_settings.database_path,
        audit_signing_key=effective_settings.audit_signing_key,
        audit_key_path=effective_settings.audit_key_path,
        timeout_seconds=effective_settings.sqlite_timeout_seconds,
    )
    artifacts = ArtifactService(
        effective_settings,
        evidence_repository,
        compiler=compiler,
    )

    app = FastAPI(
        title="EU AI Act Regulatory Evidence Pack Compiler",
        version=API_VERSION,
        description=CONSTITUTIONAL_RULE,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = effective_settings
    app.state.repository = evidence_repository
    app.state.artifact_service = artifacts
    app.state.started_at = _utc_now()
    app.state.compiler_id = COMPILER_ID
    app.state.compiler_version = COMPILER_VERSION

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        request_id = str(uuid.uuid4())
        nonce = secrets.token_urlsafe(18)
        request.state.request_id = request_id
        request.state.csp_nonce = nonce
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path.startswith("/api/") or request.url.path.startswith("/exhibit/"):
            response.headers["Cache-Control"] = "no-store"
        if response.media_type == "text/html":
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; "
                f"script-src 'nonce-{nonce}'; "
                f"style-src 'nonce-{nonce}'; "
                "connect-src 'self'; img-src 'self' data:; "
                "font-src 'none'; object-src 'none'; base-uri 'none'; "
                "form-action 'self'; frame-ancestors 'none'"
            )
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(request: Request, error: ApiProblem) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_error_body(
                request,
                error.code,
                error.message,
                error.details or None,
            ),
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {
                "location": [str(part) for part in item.get("loc", ())],
                "message": str(item.get("msg", "invalid value")),
                "type": str(item.get("type", "validation_error")),
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_body(
                request,
                "request_validation_failed",
                "The request parameters or body are invalid.",
                details,
            ),
        )

    @app.exception_handler(RepositoryConflictError)
    async def repository_conflict_handler(
        request: Request,
        error: RepositoryConflictError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=_error_body(request, "evidence_conflict", str(error)),
        )

    @app.exception_handler(RepositoryIntegrityError)
    async def repository_integrity_handler(
        request: Request,
        error: RepositoryIntegrityError,
    ) -> JSONResponse:
        logger.error("Repository integrity failure", exc_info=error)
        return JSONResponse(
            status_code=503,
            content=_error_body(
                request,
                "evidence_store_unavailable",
                "The evidence store failed an integrity check; export has been halted.",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        message = str(error.detail) if error.detail is not None else "HTTP request failed."
        return JSONResponse(
            status_code=error.status_code,
            content=_error_body(request, "http_error", message),
            headers=error.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled API failure request_id=%s",
            getattr(request.state, "request_id", "unavailable"),
            exc_info=error,
        )
        return JSONResponse(
            status_code=500,
            content=_error_body(
                request,
                "internal_error",
                "The service could not complete the request. No compliance claim was made.",
            ),
        )

    async def get_job_or_404(job_id: str) -> JobRecord:
        job = await asyncio.to_thread(evidence_repository.get_job, job_id)
        if job is None:
            raise ApiProblem(404, "job_not_found", "The requested evidence job does not exist.")
        return job

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(request: Request) -> HTMLResponse:
        jobs, _total = await asyncio.to_thread(evidence_repository.list_jobs, limit=50, offset=0)
        context = {
            "jobs": [_public_job(job, include_spans=False, include_evidence=False) for job in jobs],
            "nonce": request.state.csp_nonce,
            "api_version": API_VERSION,
            "review_key_fingerprint": effective_settings.review_key_fingerprint,
            "constitutional_rule": CONSTITUTIONAL_RULE,
        }
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context=context,
        )

    @app.get("/health", tags=["operations"])
    async def health() -> JSONResponse:
        healthy = await asyncio.to_thread(evidence_repository.health)
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "service": "exhibit-evidence-compiler",
                "version": API_VERSION,
                "compiler": {
                    "id": COMPILER_ID,
                    "version": COMPILER_VERSION,
                },
                "database": "ok" if healthy else "integrity-check-failed",
                "review_key_configured": effective_settings.review_public_key is not None,
                "time": _utc_now().isoformat(),
                "constitutional_rule": CONSTITUTIONAL_RULE,
            },
        )

    @app.post("/api/v1/ingest", status_code=201, tags=["evidence"])
    async def ingest(
        request: Request,
        trace_id: str | None = Query(default=None, min_length=1, max_length=256),
        source_name: str | None = Query(default=None, max_length=MAX_SOURCE_NAME_LENGTH),
    ) -> JSONResponse:
        if request.headers.get("content-encoding", "identity").casefold() != "identity":
            raise ApiProblem(415, "content_encoding_unsupported", "Compressed request bodies are not accepted.")
        media_type = _content_type(request)
        if media_type == "multipart/form-data":
            content, uploaded_name = await _read_multipart_upload(
                request,
                effective_settings.max_upload_bytes,
            )
            resolved_source_name = _safe_source_name(source_name or uploaded_name)
        elif media_type in {
            "application/json",
            "application/x-ndjson",
            "application/jsonl",
            "application/ndjson",
            "text/json",
            "text/plain",
        }:
            content = await _read_limited_body(request, effective_settings.max_upload_bytes)
            resolved_source_name = _safe_source_name(source_name, "trace.jsonl")
        else:
            raise ApiProblem(
                415,
                "unsupported_media_type",
                "Evidence must be supplied as JSON, JSONL, NDJSON, or a single multipart file.",
            )
        if not content:
            raise ApiProblem(422, "empty_upload", "The evidence upload is empty.")

        idempotency_key = request.headers.get("idempotency-key")
        if idempotency_key is not None and (
            not idempotency_key
            or len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LENGTH
            or not _IDEMPOTENCY_RE.fullmatch(idempotency_key)
        ):
            raise ApiProblem(
                422,
                "invalid_idempotency_key",
                "Idempotency-Key must contain 1 to 128 visible ASCII characters.",
            )

        results, warnings, ingestion = await asyncio.to_thread(
            _ingest_and_persist,
            content=content,
            source_name=resolved_source_name,
            requested_trace_id=trace_id,
            idempotency_key=idempotency_key,
            settings=effective_settings,
            repository=evidence_repository,
        )
        first = results[0].job
        items = [
            _public_job(result.job, include_spans=False, include_evidence=True)
            for result in results
        ]
        response_body: dict[str, Any] = {
            **items[0],
            "count": len(items),
            "items": items,
            "source_sha256": first.source_sha256,
            "parser_issues": [
                {
                    "source": issue.source,
                    "record": issue.record,
                    "code": issue.code,
                    "reason": issue.reason[:300],
                }
                for issue in ingestion.issues
            ],
            "warnings": warnings,
        }
        status_code = 201 if any(result.created for result in results) else 200
        return JSONResponse(
            status_code=status_code,
            content=response_body,
            headers={
                "Location": f"/exhibit/{first.job_id}",
                "X-Trace-SHA256": first.trace_sha256,
                "X-Evaluation-ID": first.evaluation_id,
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/v1/traces", tags=["evidence"])
    async def list_traces(
        job_id: str | None = Query(default=None, max_length=64),
        trace_id: str | None = Query(default=None, max_length=256),
        limit: int = Query(default=50, ge=1, le=MAX_LIST_PAGE_SIZE),
        offset: int = Query(default=0, ge=0),
        include_spans: bool = Query(default=False),
        include_evidence: bool = Query(default=False),
    ) -> JSONResponse:
        jobs, total = await asyncio.to_thread(
            evidence_repository.list_jobs,
            job_id=job_id,
            trace_id=trace_id,
            limit=limit,
            offset=offset,
        )
        if (job_id or trace_id) and not jobs:
            raise ApiProblem(404, "trace_not_found", "No evidence job matches the supplied filter.")
        return JSONResponse(
            {
                "items": [
                    _public_job(
                        job,
                        include_spans=include_spans,
                        include_evidence=include_evidence,
                    )
                    for job in jobs
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
                "constitutional_rule": CONSTITUTIONAL_RULE,
            }
        )

    @app.get("/api/v1/traces/{trace_id}", tags=["evidence"])
    async def get_trace(
        trace_id: str,
        include_spans: bool = Query(default=True),
        include_evidence: bool = Query(default=True),
    ) -> JSONResponse:
        jobs, _total = await asyncio.to_thread(
            evidence_repository.list_jobs,
            trace_id=trace_id,
            limit=MAX_LIST_PAGE_SIZE,
            offset=0,
        )
        if not jobs:
            raise ApiProblem(404, "trace_not_found", "The requested trace does not exist.")
        selected = jobs[0]
        return JSONResponse(
            {
                **_public_job(
                    selected,
                    include_spans=include_spans,
                    include_evidence=include_evidence,
                ),
                "related_job_ids": [job.job_id for job in jobs],
            }
        )

    @app.post("/api/v1/review", tags=["governance"])
    async def submit_review(request: Request) -> JSONResponse:
        if _content_type(request) != "application/json":
            raise ApiProblem(
                415,
                "application_json_required",
                "Human review submissions must use application/json.",
            )
        if request.headers.get("content-encoding", "identity").casefold() != "identity":
            raise ApiProblem(415, "content_encoding_unsupported", "Compressed request bodies are not accepted.")
        raw = await _read_limited_body(request, effective_settings.max_upload_bytes)
        try:
            document = _strict_json_loads(raw.decode("utf-8-sig", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ApiProblem(422, "invalid_review_json", "The review submission is not valid UTF-8 JSON.") from exc
        if not isinstance(document, Mapping):
            raise ApiProblem(422, "invalid_review", "The review submission must be a JSON object.")

        signoff_payload: Any = document
        for envelope_key in ("signoff", "review"):
            if envelope_key in document:
                signoff_payload = document[envelope_key]
                break
        else:
            if isinstance(document.get("payload"), Mapping) and document.get("signature"):
                signoff_payload = {**document["payload"], "signature": document["signature"]}
        if not isinstance(signoff_payload, Mapping):
            raise ApiProblem(
                422,
                "invalid_review",
                "The signoff or review property must contain a signed HumanReviewSignoff object.",
            )

        review_job_id = signoff_payload.get("job_id")
        if not isinstance(review_job_id, str):
            raise ApiProblem(422, "review_job_required", "The signed review must identify a job_id.")
        job = await get_job_or_404(review_job_id)
        if effective_settings.review_public_key is None:
            raise ApiProblem(
                503,
                "review_key_not_configured",
                "Human review verification is unavailable until an operator configures the trusted reviewer public key.",
            )
        try:
            review = HumanReviewSignoff.model_validate(signoff_payload)
        except ValidationError as exc:
            details = [
                {
                    "location": [str(part) for part in item["loc"]],
                    "message": item["msg"],
                    "type": item["type"],
                }
                for item in exc.errors(include_input=False, include_url=False)
            ]
            raise ApiProblem(
                422,
                "invalid_review_schema",
                "The human-review sign-off does not satisfy the required schema.",
                details,
            ) from exc
        _validate_review_binding(review, job)
        try:
            verify_signed_human_review(review, effective_settings.review_public_key)
        except InvalidReviewSignatureError as exc:
            raise ApiProblem(
                422,
                "review_signature_invalid",
                "The human-review signature could not be verified.",
            ) from exc
        except ReviewVerificationConfigurationError as exc:
            raise ApiProblem(
                503,
                "review_verifier_unavailable",
                str(exc),
            ) from exc

        try:
            result = await asyncio.to_thread(
                evidence_repository.record_review,
                job_id=job.job_id,
                review=review,
                public_key=effective_settings.review_public_key,
            )
        except RepositoryConflictError as exc:
            raise ApiProblem(409, "review_conflict", str(exc)) from exc
        response = _public_job(
            result.job,
            include_spans=False,
            include_evidence=False,
        )
        return JSONResponse(
            status_code=201 if result.created else 200,
            content={
                **response,
                "review_verification": {
                    "status": "verified",
                    "actor_type": "human",
                    "reviewer_id": _model_value(
                        review,
                        "reviewer_id",
                        "actor_id",
                        "human_reviewer_id",
                        "reviewer_name",
                    ),
                    "decision": _model_value(review, "decision", "review_decision"),
                    "key_fingerprint": effective_settings.review_key_fingerprint,
                    "receipt_event": "human.review.signed",
                },
            },
            headers={
                "Location": f"/exhibit/{result.job.job_id}",
                "Cache-Control": "no-store",
            },
        )

    async def materialize_exhibit(job_id: str) -> tuple[JobRecord, ExportResult, bytes]:
        job = await get_job_or_404(job_id)
        try:
            export_result = await asyncio.to_thread(artifacts.export_job, job)
            _path, content, _digest = await asyncio.to_thread(artifacts.read_artifact, job)
            refreshed = await asyncio.to_thread(evidence_repository.get_job, job_id)
            if refreshed is None:
                raise RepositoryIntegrityError("job disappeared during export")
            return refreshed, export_result, content
        except RepositoryConflictError as exc:
            raise ApiProblem(409, "export_not_authorized", str(exc)) from exc
        except ExportNotAuthorized as exc:
            raise ApiProblem(
                409,
                "export_policy_denied",
                "The governance policy denied exhibit export.",
                _policy_decision_details(exc),
            ) from exc
        except HumanReviewRequired as exc:
            raise ApiProblem(
                409,
                "human_review_required",
                "A verified, approved human review is required before exhibit export.",
            ) from exc
        except InvalidPolicyHumanReviewError as exc:
            raise ApiProblem(
                422,
                "invalid_human_review",
                "The persisted human review is invalid.",
            ) from exc
        except PolicyBindingError as exc:
            raise ApiProblem(
                409,
                "policy_binding_mismatch",
                str(exc),
            ) from exc
        except AutomatedComplianceStampProhibited as exc:
            raise ApiProblem(
                403,
                "automated_stamp_prohibited",
                str(exc),
            ) from exc
        except PolicyError as exc:
            raise ApiProblem(409, "export_policy_denied", str(exc)) from exc
        except CompilationError as exc:
            raise ApiProblem(422, "compilation_failed", str(exc)) from exc

    @app.get("/exhibit/{job_id}", tags=["exports"])
    @app.head("/exhibit/{job_id}", tags=["exports"])
    async def export_exhibit(job_id: str, request: Request) -> Response:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ApiProblem(404, "job_not_found", "The requested evidence job does not exist.")
        job, result, content = await materialize_exhibit(job_id)
        etag = f'"{result.sha256}"'
        headers = {
            "ETag": etag,
            "Content-Disposition": 'attachment; filename="exhibit.json"',
            "X-Exhibit-SHA256": result.sha256,
            "X-Trace-SHA256": job.trace_sha256,
            "X-Evaluation-ID": job.evaluation_id,
            "Digest": f"sha-256={base64.b64encode(bytes.fromhex(result.sha256)).decode('ascii')}",
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(
            content=content,
            status_code=200,
            media_type=PACK_MEDIA_TYPE,
            headers=headers,
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    return app


__all__ = [
    "API_VERSION",
    "ApiProblem",
    "ArtifactService",
    "CreateJobResult",
    "EvidenceRepository",
    "ExportResult",
    "JobRecord",
    "RepositoryConflictError",
    "RepositoryError",
    "RepositoryIntegrityError",
    "ReviewResult",
    "Settings",
    "create_app",
    "verify_signed_human_review",
]