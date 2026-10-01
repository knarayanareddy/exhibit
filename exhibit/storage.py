from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import stat
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .models import (
    CONSTITUTIONAL_RULE,
    UTC,
    HumanReviewSignoff,
    Trace,
    TraceFormat,
    _freeze_value,
    _normalise_event_type,
    _normalise_hash,
    _normalise_identifier,
    _normalise_timestamp,
    canonical_json_bytes,
    compute_trace_sha256,
    sha256_hex,
    verify_human_review_bytes,
)


SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
_RECEIPT_HASH_CONTEXT = b"exhibit/audit-receipt-chain/v1\x00"
_RECEIPT_SIGNATURE_CONTEXT = b"exhibit/audit-receipt-signature/v1\x00"
_GENESIS_HASH = "0" * 64
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HUMAN_REVIEW_EVENTS = {
    "human.review.signed",
    "human.review.signoff",
    "review.signed",
}


class StorageError(RuntimeError):
    """Base class for persistent evidence-store failures."""


class StorageConfigurationError(StorageError):
    """The receipt store has unsafe or invalid configuration."""


class ReceiptNotFoundError(StorageError, LookupError):
    """A requested audit receipt does not exist."""


class EvaluationNotFoundError(StorageError, LookupError):
    """A requested evaluation result does not exist."""


class HumanReviewNotFoundError(StorageError, LookupError):
    """A requested human-review sign-off does not exist."""


class TraceRegistrationNotFoundError(StorageError, LookupError):
    """A requested trace registration does not exist."""


class ReceiptConflictError(StorageError):
    """An immutable identifier or idempotency key was reused."""


class ReceiptChainError(StorageError):
    """The append-only receipt chain failed integrity verification."""


class InvalidHumanReviewError(StorageError, ValueError):
    """A purported human sign-off failed identity, actor, or signature checks."""


class ReceiptActorType(str, Enum):
    HUMAN = "human"
    SERVICE = "service"
    EVALUATOR = "evaluator"

    @classmethod
    def _missing_(cls, value: object) -> ReceiptActorType | None:
        if not isinstance(value, str):
            return None
        normalised = value.strip().casefold().replace("_", "-")
        aliases = {
            "human-reviewer": cls.HUMAN,
            "reviewer": cls.HUMAN,
            "operator": cls.HUMAN,
            "system": cls.SERVICE,
            "agent": cls.SERVICE,
            "service-account": cls.SERVICE,
            "evaluation-engine": cls.EVALUATOR,
            "evaluation-service": cls.EVALUATOR,
        }
        if normalised in {item.value for item in cls}:
            return cls(normalised)
        return aliases.get(normalised)


class _StorageModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
    )


class AuditReceipt(_StorageModel):
    receipt_id: str
    sequence: int = Field(ge=1)
    schema_version: int = SCHEMA_VERSION
    event_type: str
    actor_type: ReceiptActorType
    actor_id: str
    job_id: str | None = None
    trace_id: str | None = None
    occurred_at: datetime
    payload: Mapping[str, Any]
    payload_hash: str = Field(validation_alias=AliasChoices("payload_hash", "payload_sha256"))
    previous_receipt_hash: str | None = None
    receipt_hash: str = Field(
        validation_alias=AliasChoices("receipt_hash", "receipt_sha256", "chain_hash")
    )
    signature: str
    signing_key_id: str
    idempotency_key: str | None = None

    @field_validator("receipt_id", "actor_id", mode="before")
    @classmethod
    def normalise_required_identifier(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, required=True)
        assert identifier is not None
        return identifier

    @field_validator("job_id", "trace_id", mode="before")
    @classmethod
    def normalise_optional_identifier(cls, value: Any) -> str | None:
        return _normalise_identifier(value)

    @field_validator("event_type", mode="before")
    @classmethod
    def normalise_event_type(cls, value: Any) -> str:
        return _normalise_event_type(value)

    @field_validator("occurred_at", mode="before")
    @classmethod
    def normalise_occurred_at(cls, value: Any) -> datetime:
        timestamp = _normalise_timestamp(value, required=True)
        assert timestamp is not None
        return timestamp

    @field_validator("payload", mode="before")
    @classmethod
    def normalise_payload(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("receipt payload must be a mapping")
        if len(canonical_json_bytes(value)) > MAX_PAYLOAD_BYTES:
            raise ValueError("receipt payload exceeds the configured size limit")
        return value

    @field_validator("payload_hash", "previous_receipt_hash", "receipt_hash", mode="before")
    @classmethod
    def normalise_digest(cls, value: Any) -> str | None:
        return _normalise_hash(value)

    @field_validator("signing_key_id", mode="before")
    @classmethod
    def normalise_key_id(cls, value: Any) -> str:
        key_id = str(value).strip()
        if not _KEY_ID_RE.fullmatch(key_id):
            raise ValueError("invalid receipt signing key ID")
        return key_id

    @field_validator("idempotency_key", mode="before")
    @classmethod
    def normalise_idempotency_key(cls, value: Any) -> str | None:
        return _normalise_identifier(value)

    @field_validator("signature", mode="before")
    @classmethod
    def validate_signature(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("receipt signature must be base64url text")
        try:
            decoded = base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as error:
            raise ValueError("receipt signature is not valid base64url") from error
        if len(decoded) != hashlib.sha256().digest_size:
            raise ValueError("receipt signature must be a SHA-256 HMAC")
        return value

    @model_validator(mode="after")
    def freeze_payload(self) -> AuditReceipt:
        object.__setattr__(self, "payload", _freeze_value(self.payload))
        return self

    @property
    def receipt_sha256(self) -> str:
        return self.receipt_hash

    @property
    def payload_sha256(self) -> str:
        return self.payload_hash

    @property
    def timestamp(self) -> datetime:
        return self.occurred_at

    @property
    def created_at(self) -> datetime:
        return self.occurred_at

    def verify_signature(self, key: bytes) -> bool:
        return _verify_receipt_signature(self, key)


class TraceRegistration(_StorageModel):
    job_id: str
    trace_id: str
    trace_sha256: str
    source_sha256: str | None = None
    trace_format: TraceFormat = TraceFormat.OPENINFERENCE
    span_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    receipt_id: str
    registered_at: datetime
    registration_sha256: str

    @property
    def source_hash(self) -> str | None:
        return self.source_sha256

    @property
    def trace_hash(self) -> str:
        return self.trace_sha256


class StoredEvaluation(_StorageModel):
    evaluation_id: str
    job_id: str
    trace_id: str
    trace_sha256: str
    payload: Mapping[str, Any]
    payload_sha256: str
    receipt_id: str
    stored_at: datetime

    @field_validator("payload", mode="before")
    @classmethod
    def normalise_payload(cls, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("evaluation payload must be a mapping")
        return value

    @model_validator(mode="after")
    def freeze_payload(self) -> StoredEvaluation:
        object.__setattr__(self, "payload", _freeze_value(self.payload))
        return self

    @property
    def data(self) -> Mapping[str, Any]:
        return self.payload

    @property
    def result(self) -> Mapping[str, Any]:
        return self.payload


class HumanReviewRecord(_StorageModel):
    signoff_id: str
    job_id: str
    trace_id: str
    trace_sha256: str
    signoff: HumanReviewSignoff
    signoff_sha256: str
    receipt_id: str
    recorded_at: datetime

    @property
    def payload(self) -> HumanReviewSignoff:
        return self.signoff


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key '{key}'")
        result[key] = value
    return result


def _json_loads(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_reject_duplicate_json_keys)


def _json_object(value: Any) -> dict[str, Any]:
    parsed = _json_loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stored canonical JSON must contain an object")
    return parsed


def _format_timestamp(value: datetime) -> str:
    timestamp = _normalise_timestamp(value, required=True)
    assert timestamp is not None
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalise_key(key: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise StorageConfigurationError("HMAC signing keys must be bytes")
    value = bytes(key)
    if len(value) < 32:
        raise StorageConfigurationError("HMAC signing keys must contain at least 32 bytes")
    return value


def _normalise_key_id(value: Any) -> str:
    key_id = str(value).strip()
    if not _KEY_ID_RE.fullmatch(key_id):
        raise StorageConfigurationError("invalid signing key ID")
    return key_id


def _receipt_body(
    receipt: AuditReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(receipt, AuditReceipt):
        values = receipt.model_dump(mode="python")
    elif isinstance(receipt, Mapping):
        values = dict(receipt)
    else:
        raise TypeError("receipt body requires an AuditReceipt or mapping")

    actor_type = values.get("actor_type")
    if isinstance(actor_type, Enum):
        actor_type = actor_type.value
    occurred_at = values.get("occurred_at")
    if isinstance(occurred_at, datetime):
        occurred_at = _format_timestamp(occurred_at)
    elif isinstance(occurred_at, str):
        timestamp = _normalise_timestamp(occurred_at, required=True)
        assert timestamp is not None
        occurred_at = _format_timestamp(timestamp)

    return {
        "schema_version": int(values.get("schema_version", SCHEMA_VERSION)),
        "sequence": int(values["sequence"]),
        "receipt_id": values["receipt_id"],
        "event_type": values["event_type"],
        "actor_type": str(actor_type),
        "actor_id": values["actor_id"],
        "job_id": values.get("job_id"),
        "trace_id": values.get("trace_id"),
        "occurred_at": occurred_at,
        "payload": values.get("payload", {}),
        "payload_hash": values["payload_hash"],
        "previous_receipt_hash": values.get("previous_receipt_hash"),
        "receipt_hash": values["receipt_hash"],
    }


def _calculate_receipt_hash(receipt: AuditReceipt | Mapping[str, Any]) -> str:
    return sha256_hex(_RECEIPT_HASH_CONTEXT + canonical_json_bytes(_receipt_body(receipt)))


def _receipt_signature(receipt: AuditReceipt | Mapping[str, Any], key: bytes) -> str:
    signing_key = _normalise_key(key)
    body = canonical_json_bytes(_receipt_body(receipt))
    digest = hmac.new(
        signing_key,
        _RECEIPT_SIGNATURE_CONTEXT + body,
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


def _verify_receipt_signature(receipt: AuditReceipt, key: bytes) -> bool:
    try:
        supplied = base64.b64decode(
            receipt.signature.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        expected = base64.urlsafe_b64decode(
            _receipt_signature(receipt, key).encode("ascii")
        )
    except (ValueError, UnicodeEncodeError):
        return False
    return len(supplied) == len(expected) and hmac.compare_digest(supplied, expected)


def _registration_payload(registration: TraceRegistration | Mapping[str, Any]) -> dict[str, Any]:
    values = (
        registration.model_dump(mode="python")
        if isinstance(registration, TraceRegistration)
        else dict(registration)
    )
    registered_at = values.get("registered_at")
    if isinstance(registered_at, datetime):
        values["registered_at"] = _format_timestamp(registered_at)
    trace_format = values.get("trace_format")
    if isinstance(trace_format, Enum):
        values["trace_format"] = trace_format.value
    return values


def _review_payload(review: HumanReviewSignoff) -> dict[str, Any]:
    return review.model_dump(mode="json")


class ReceiptsStore:
    """Append-only SQLite evidence store with a signed receipt hash chain."""

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        signing_key: bytes | bytearray | memoryview | None = None,
        *,
        key: bytes | bytearray | memoryview | None = None,
        key_id: str | None = None,
        signing_key_id: str | None = None,
        trusted_keys: Mapping[str, bytes | bytearray | memoryview]
        | tuple[bytes | bytearray | memoryview, ...]
        | None = None,
        key_resolver: Callable[[str], bytes | bytearray | memoryview | None] | None = None,
        busy_timeout_ms: int = 5_000,
        strict_permissions: bool = True,
        clock: Callable[[], datetime] | None = None,
        **unknown_options: Any,
    ) -> None:
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise StorageConfigurationError(f"unsupported storage options: {names}")
        if signing_key is not None and key is not None:
            raise StorageConfigurationError("supply only one of signing_key or key")
        active_key = signing_key if signing_key is not None else key
        if key_id is not None and signing_key_id is not None and key_id != signing_key_id:
            raise StorageConfigurationError("key_id and signing_key_id disagree")
        selected_key_id = key_id or signing_key_id
        if not isinstance(busy_timeout_ms, int) or isinstance(busy_timeout_ms, bool):
            raise StorageConfigurationError("busy_timeout_ms must be an integer")
        if not 0 <= busy_timeout_ms <= 300_000:
            raise StorageConfigurationError("busy_timeout_ms must be between 0 and 300000")
        if key_resolver is not None and not callable(key_resolver):
            raise StorageConfigurationError("key_resolver must be callable")
        if clock is not None and not callable(clock):
            raise StorageConfigurationError("clock must be callable")

        self._memory_database = str(database_path) == ":memory:"
        self.database_path = Path(database_path)
        self._closed = False
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._key_resolver = key_resolver
        self._strict_permissions = bool(strict_permissions)

        self._trusted_keys: dict[str, bytes] = {}
        if trusted_keys is not None:
            if isinstance(trusted_keys, Mapping):
                for raw_key_id, raw_key in trusted_keys.items():
                    self._trusted_keys[_normalise_key_id(raw_key_id)] = _normalise_key(raw_key)
            else:
                for index, raw_key in enumerate(trusted_keys):
                    key_name = selected_key_id if selected_key_id and index == 0 else f"trusted-{index}"
                    self._trusted_keys[_normalise_key_id(key_name)] = _normalise_key(raw_key)

        self.ephemeral_signing_key = active_key is None and not self._trusted_keys
        if self.ephemeral_signing_key:
            self._signing_key = secrets.token_bytes(48)
            self._signing_key_id = "ephemeral"
        else:
            if active_key is None:
                if selected_key_id is None:
                    if len(self._trusted_keys) != 1:
                        raise StorageConfigurationError(
                            "signing_key_id is required when multiple trusted keys are supplied"
                        )
                    selected_key_id = next(iter(self._trusted_keys))
                try:
                    active_key = self._trusted_keys[selected_key_id]
                except KeyError as error:
                    raise StorageConfigurationError(
                        "the active signing key ID is not present in trusted_keys"
                    ) from error
            self._signing_key = _normalise_key(active_key)
            self._signing_key_id = _normalise_key_id(selected_key_id or "primary")
        self._trusted_keys[self._signing_key_id] = self._signing_key

        self._prepare_database_path()
        try:
            self._connection = sqlite3.connect(
                str(database_path),
                timeout=busy_timeout_ms / 1000,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error as error:
            raise StorageConfigurationError(f"unable to open receipt database: {error}") from error
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure_connection(busy_timeout_ms)
            self._initialise_schema()
            if not self._memory_database:
                os.chmod(self.database_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    @property
    def signing_key_id(self) -> str:
        return self._signing_key_id

    @property
    def trusted_key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._trusted_keys))

    @property
    def is_closed(self) -> bool:
        return self._closed

    def export_active_signing_key(self) -> bytes:
        """Return the active secret for controlled key escrow or rotation."""

        self._ensure_open()
        return self._signing_key

    def _prepare_database_path(self) -> None:
        if self._memory_database:
            return
        path = Path(self.database_path)
        try:
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            raise StorageConfigurationError(
                f"unable to create receipt database directory: {error}"
            ) from error

        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                    stat.S_IRUSR | stat.S_IWUSR,
                )
            except OSError as error:
                raise StorageConfigurationError(
                    f"unable to create receipt database: {error}"
                ) from error
            os.close(descriptor)
            return
        except OSError as error:
            raise StorageConfigurationError(f"unable to inspect receipt database: {error}") from error

        if stat.S_ISLNK(path_stat.st_mode):
            raise StorageConfigurationError("receipt database cannot be a symbolic link")
        if not stat.S_ISREG(path_stat.st_mode):
            raise StorageConfigurationError("receipt database path must be a regular file")
        permissions = stat.S_IMODE(path_stat.st_mode)
        if self._strict_permissions and permissions & 0o077:
            raise StorageConfigurationError(
                "receipt database permissions must deny group and other access"
            )

    def _configure_connection(self, busy_timeout_ms: int) -> None:
        cursor = self._connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            cursor.execute("PRAGMA journal_mode = DELETE")
            cursor.execute("PRAGMA synchronous = FULL")
            cursor.execute("PRAGMA trusted_schema = OFF")
        except sqlite3.Error as error:
            raise StorageConfigurationError(f"unable to configure receipt database: {error}") from error
        finally:
            cursor.close()

    def _initialise_schema(self) -> None:
        created_at = _format_timestamp(self._now())
        statements = (
            """
            CREATE TABLE IF NOT EXISTS exhibit_schema_metadata (
                metadata_key TEXT PRIMARY KEY,
                metadata_value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS receipts (
                sequence INTEGER PRIMARY KEY CHECK (sequence >= 1),
                receipt_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                actor_type TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                job_id TEXT,
                trace_id TEXT,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                previous_receipt_hash TEXT,
                receipt_hash TEXT NOT NULL UNIQUE,
                receipt_body_json TEXT NOT NULL,
                signature TEXT NOT NULL,
                signing_key_id TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                schema_version INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trace_registrations (
                job_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                trace_sha256 TEXT NOT NULL,
                source_sha256 TEXT,
                trace_format TEXT NOT NULL,
                span_count INTEGER NOT NULL CHECK (span_count >= 0),
                event_count INTEGER NOT NULL CHECK (event_count >= 0),
                trace_json TEXT NOT NULL,
                registration_sha256 TEXT NOT NULL,
                receipt_id TEXT NOT NULL UNIQUE,
                registered_at TEXT NOT NULL,
                PRIMARY KEY (job_id, trace_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                trace_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                receipt_id TEXT NOT NULL UNIQUE,
                stored_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, job_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_reviews (
                signoff_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                trace_sha256 TEXT NOT NULL,
                signoff_json TEXT NOT NULL,
                signoff_sha256 TEXT NOT NULL,
                receipt_id TEXT NOT NULL UNIQUE,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (signoff_id, job_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS receipts_job_idx ON receipts(job_id, sequence)",
            "CREATE INDEX IF NOT EXISTS receipts_trace_idx ON receipts(trace_id, sequence)",
            "CREATE INDEX IF NOT EXISTS evaluations_job_idx ON evaluations(job_id, stored_at)",
            "CREATE INDEX IF NOT EXISTS human_reviews_job_idx ON human_reviews(job_id, recorded_at)",
            "CREATE VIEW IF NOT EXISTS audit_receipts AS SELECT * FROM receipts",
        )
        triggers = (
            ("exhibit_schema_metadata_no_update", "exhibit_schema_metadata"),
            ("exhibit_schema_metadata_no_delete", "exhibit_schema_metadata"),
            ("receipts_no_update", "receipts"),
            ("receipts_no_delete", "receipts"),
            ("trace_registrations_no_update", "trace_registrations"),
            ("trace_registrations_no_delete", "trace_registrations"),
            ("evaluations_no_update", "evaluations"),
            ("evaluations_no_delete", "evaluations"),
            ("human_reviews_no_update", "human_reviews"),
            ("human_reviews_no_delete", "human_reviews"),
        )
        try:
            self._connection.execute("BEGIN EXCLUSIVE")
            for statement in statements:
                self._connection.execute(statement)

            expected_metadata = {
                "schema_version": str(SCHEMA_VERSION),
                "created_at": created_at,
                "canonical_json": "utf8-json-sort-keys-v1",
                "receipt_hash_algorithm": "sha256",
                "receipt_signature_algorithm": "hmac-sha256",
                "constitutional_rule": CONSTITUTIONAL_RULE,
            }
            for metadata_key, metadata_value in expected_metadata.items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO exhibit_schema_metadata "
                    "(metadata_key, metadata_value) VALUES (?, ?)",
                    (metadata_key, metadata_value),
                )
            rows = self._connection.execute(
                "SELECT metadata_key, metadata_value FROM exhibit_schema_metadata"
            ).fetchall()
            existing_metadata = {row["metadata_key"]: row["metadata_value"] for row in rows}
            for metadata_key, expected_value in expected_metadata.items():
                if existing_metadata.get(metadata_key) != expected_value:
                    raise StorageConfigurationError(
                        f"receipt database metadata '{metadata_key}' is incompatible"
                    )
            if set(existing_metadata) != set(expected_metadata):
                raise StorageConfigurationError("receipt database metadata keys are unexpected")

            for operation in ("UPDATE", "DELETE"):
                for trigger_name, table_name in triggers:
                    trigger_sql = f"""
                        CREATE TRIGGER IF NOT EXISTS {trigger_name}
                        BEFORE {operation} ON {table_name}
                        BEGIN
                            SELECT RAISE(ABORT, 'immutable evidence');
                        END
                    """
                    self._connection.execute(trigger_sql)
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("receipt store is closed")

    def _now(self) -> datetime:
        value = self._clock()
        timestamp = _normalise_timestamp(value, required=True)
        assert timestamp is not None
        return timestamp

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        self._ensure_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as error:
                raise StorageError(f"unable to begin receipt transaction: {error}") from error
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _resolve_key(self, key_id: str) -> bytes:
        key = self._trusted_keys.get(key_id)
        if key is None and self._key_resolver is not None:
            resolved = self._key_resolver(key_id)
            if resolved is not None:
                key = _normalise_key(resolved)
                self._trusted_keys[key_id] = key
        if key is None:
            raise ReceiptChainError(f"no verification key is available for key ID '{key_id}'")
        return key

    @staticmethod
    def _row_receipt(row: sqlite3.Row) -> AuditReceipt:
        return AuditReceipt.model_validate(
            {
                "schema_version": row["schema_version"],
                "sequence": row["sequence"],
                "receipt_id": row["receipt_id"],
                "event_type": row["event_type"],
                "actor_type": row["actor_type"],
                "actor_id": row["actor_id"],
                "job_id": row["job_id"],
                "trace_id": row["trace_id"],
                "occurred_at": row["occurred_at"],
                "payload": _json_object(row["payload_json"]),
                "payload_hash": row["payload_hash"],
                "previous_receipt_hash": row["previous_receipt_hash"],
                "receipt_hash": row["receipt_hash"],
                "signature": row["signature"],
                "signing_key_id": row["signing_key_id"],
                "idempotency_key": row["idempotency_key"],
            }
        )

    @staticmethod
    def _receipt_matches(
        receipt: AuditReceipt,
        *,
        event_type: str,
        actor_type: ReceiptActorType,
        actor_id: str,
        job_id: str | None,
        trace_id: str | None,
        payload_hash: str,
        idempotency_key: str | None,
        occurred_at: datetime | None,
    ) -> bool:
        return (
            receipt.event_type == event_type
            and receipt.actor_type == actor_type
            and receipt.actor_id == actor_id
            and receipt.job_id == job_id
            and receipt.trace_id == trace_id
            and receipt.payload_hash == payload_hash
            and receipt.idempotency_key == idempotency_key
            and (occurred_at is None or receipt.occurred_at == occurred_at)
        )

    def _validate_human_receipt_payload(
        self,
        event_type: str,
        actor_type: ReceiptActorType,
        actor_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        is_review_event = event_type in _HUMAN_REVIEW_EVENTS
        if actor_type == ReceiptActorType.HUMAN and not is_review_event:
            raise InvalidHumanReviewError(
                "actor='human' is permitted only for a cryptographic review sign-off event"
            )
        if is_review_event and actor_type != ReceiptActorType.HUMAN:
            raise InvalidHumanReviewError("a review-signoff receipt must have actor='human'")
        if not is_review_event:
            return

        raw_signoff = payload.get("signoff", payload)
        try:
            signoff = (
                raw_signoff
                if isinstance(raw_signoff, HumanReviewSignoff)
                else HumanReviewSignoff.model_validate(raw_signoff)
            )
        except Exception as error:
            raise InvalidHumanReviewError("human-review receipt does not contain a valid sign-off") from error
        if signoff.actor != "human":
            raise InvalidHumanReviewError("a human-review sign-off must have actor='human'")
        if signoff.reviewer_id != actor_id:
            raise InvalidHumanReviewError("receipt actor_id does not match the human reviewer")
        try:
            key = self._resolve_key(signoff.signing_key_id)
        except ReceiptChainError as error:
            raise InvalidHumanReviewError(
                f"no trusted key is available for review key ID '{signoff.signing_key_id}'"
            ) from error
        if not verify_human_review_bytes(signoff.signing_payload, key, signoff.signature):
            raise InvalidHumanReviewError("human-review signature verification failed")

    def _insert_receipt_locked(
        self,
        connection: sqlite3.Connection,
        *,
        receipt_id: str | None,
        event_type: Any,
        actor_type: Any,
        actor_id: Any,
        job_id: Any,
        trace_id: Any,
        payload: Mapping[str, Any] | None,
        occurred_at: Any,
        idempotency_key: Any,
    ) -> AuditReceipt:
        normalised_event = _normalise_event_type(event_type)
        normalised_actor = ReceiptActorType(actor_type)
        normalised_actor_id = _normalise_identifier(actor_id, required=True)
        assert normalised_actor_id is not None
        normalised_job_id = _normalise_identifier(job_id)
        normalised_trace_id = _normalise_identifier(trace_id)
        normalised_idempotency_key = _normalise_identifier(idempotency_key)
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise StorageError("receipt payload must be a mapping")
        payload_bytes = canonical_json_bytes(payload)
        if len(payload_bytes) > MAX_PAYLOAD_BYTES:
            raise StorageError("receipt payload exceeds the configured size limit")
        payload_hash = sha256_hex(payload_bytes)
        supplied_occurred_at = (
            _normalise_timestamp(occurred_at) if occurred_at is not None else None
        )

        self._validate_human_receipt_payload(
            normalised_event,
            normalised_actor,
            normalised_actor_id,
            payload,
        )

        existing_by_id: sqlite3.Row | None = None
        if receipt_id is not None:
            normalised_receipt_id = _normalise_identifier(receipt_id, required=True)
            assert normalised_receipt_id is not None
            existing_by_id = connection.execute(
                "SELECT * FROM receipts WHERE receipt_id = ?",
                (normalised_receipt_id,),
            ).fetchone()
        else:
            normalised_receipt_id = None

        existing_by_idempotency: sqlite3.Row | None = None
        if normalised_idempotency_key is not None:
            existing_by_idempotency = connection.execute(
                "SELECT * FROM receipts WHERE idempotency_key = ?",
                (normalised_idempotency_key,),
            ).fetchone()

        if existing_by_id is not None:
            existing = self._row_receipt(existing_by_id)
            if self._receipt_matches(
                existing,
                event_type=normalised_event,
                actor_type=normalised_actor,
                actor_id=normalised_actor_id,
                job_id=normalised_job_id,
                trace_id=normalised_trace_id,
                payload_hash=payload_hash,
                idempotency_key=normalised_idempotency_key,
                occurred_at=supplied_occurred_at,
            ):
                return existing
            raise ReceiptConflictError(
                "receipt_id was reused for different receipt evidence"
            )

        if existing_by_idempotency is not None:
            existing = self._row_receipt(existing_by_idempotency)
            if self._receipt_matches(
                existing,
                event_type=normalised_event,
                actor_type=normalised_actor,
                actor_id=normalised_actor_id,
                job_id=normalised_job_id,
                trace_id=normalised_trace_id,
                payload_hash=payload_hash,
                idempotency_key=normalised_idempotency_key,
                occurred_at=supplied_occurred_at,
            ):
                return existing
            raise ReceiptConflictError(
                "idempotency_key was reused for different receipt evidence"
            )

        head_row = connection.execute(
            "SELECT * FROM receipts ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = head_row["receipt_hash"] if head_row is not None else None
        sequence = int(head_row["sequence"]) + 1 if head_row is not None else 1
        effective_occurred_at = supplied_occurred_at or self._now()
        if head_row is not None:
            previous_time = _normalise_timestamp(head_row["occurred_at"], required=True)
            assert previous_time is not None and effective_occurred_at is not None
            if effective_occurred_at < previous_time:
                raise StorageError("receipt timestamps cannot move backwards")

        if normalised_receipt_id is None:
            normalised_receipt_id = f"receipt-{uuid.uuid4().hex}"

        unsigned_receipt = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "receipt_id": normalised_receipt_id,
            "event_type": normalised_event,
            "actor_type": normalised_actor,
            "actor_id": normalised_actor_id,
            "job_id": normalised_job_id,
            "trace_id": normalised_trace_id,
            "occurred_at": effective_occurred_at,
            "payload": payload,
            "payload_hash": payload_hash,
            "previous_receipt_hash": previous_hash,
        }
        receipt_hash = _calculate_receipt_hash(unsigned_receipt)
        receipt_values = {**unsigned_receipt, "receipt_hash": receipt_hash}
        signature = _receipt_signature(receipt_values, self._signing_key)
        receipt = AuditReceipt.model_validate(
            {
                **receipt_values,
                "signature": signature,
                "signing_key_id": self._signing_key_id,
                "idempotency_key": normalised_idempotency_key,
            }
        )
        body_json = canonical_json_bytes(_receipt_body(receipt)).decode("utf-8")
        connection.execute(
            """
            INSERT INTO receipts (
                sequence, receipt_id, event_type, actor_type, actor_id,
                job_id, trace_id, occurred_at, payload_json, payload_hash,
                previous_receipt_hash, receipt_hash, receipt_body_json,
                signature, signing_key_id, idempotency_key, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.sequence,
                receipt.receipt_id,
                receipt.event_type,
                receipt.actor_type.value,
                receipt.actor_id,
                receipt.job_id,
                receipt.trace_id,
                _format_timestamp(receipt.occurred_at),
                canonical_json_bytes(receipt.payload).decode("utf-8"),
                receipt.payload_hash,
                receipt.previous_receipt_hash,
                receipt.receipt_hash,
                body_json,
                receipt.signature,
                receipt.signing_key_id,
                receipt.idempotency_key,
                receipt.schema_version,
            ),
        )
        return receipt

    def append_receipt(
        self,
        event_type: str,
        actor_type: ReceiptActorType | str,
        actor_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        job_id: str | None = None,
        trace_id: str | None = None,
        receipt_id: str | None = None,
        occurred_at: datetime | str | None = None,
        idempotency_key: str | None = None,
    ) -> AuditReceipt:
        with self._write_transaction() as connection:
            return self._insert_receipt_locked(
                connection,
                receipt_id=receipt_id,
                event_type=event_type,
                actor_type=actor_type,
                actor_id=actor_id,
                job_id=job_id,
                trace_id=trace_id,
                payload=payload,
                occurred_at=occurred_at,
                idempotency_key=idempotency_key,
            )

    def record_receipt(
        self,
        event_type: str,
        actor_type: ReceiptActorType | str,
        actor_id: str,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditReceipt:
        return self.append_receipt(
            event_type,
            actor_type,
            actor_id,
            payload,
            **kwargs,
        )

    def count_receipts(self) -> int:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM receipts").fetchone()
        return int(row["count"])

    def get_receipt(self, receipt_id: str) -> AuditReceipt:
        self._ensure_open()
        identifier = _normalise_identifier(receipt_id, required=True)
        assert identifier is not None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM receipts WHERE receipt_id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise ReceiptNotFoundError(f"receipt '{identifier}' does not exist")
        return self._row_receipt(row)

    def get_chain_head(self) -> AuditReceipt | None:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        return self._row_receipt(row) if row is not None else None

    def iter_receipts(
        self,
        *,
        job_id: str | None = None,
        trace_id: str | None = None,
        event_type: str | None = None,
    ) -> Iterator[AuditReceipt]:
        self._ensure_open()
        clauses: list[str] = []
        parameters: list[str] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            parameters.append(_normalise_identifier(job_id, required=True))  # type: ignore[arg-type]
        if trace_id is not None:
            clauses.append("trace_id = ?")
            parameters.append(_normalise_identifier(trace_id, required=True))  # type: ignore[arg-type]
        if event_type is not None:
            clauses.append("event_type = ?")
            parameters.append(_normalise_event_type(event_type))
        query = "SELECT * FROM receipts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY sequence"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        for row in rows:
            yield self._row_receipt(row)

    def list_receipts(self, **filters: Any) -> tuple[AuditReceipt, ...]:
        return tuple(self.iter_receipts(**filters))

    def _verify_receipt_row(self, row: sqlite3.Row, previous_hash: str | None) -> bool:
        try:
            receipt = self._row_receipt(row)
            if receipt.sequence != int(row["sequence"]):
                return False
            if receipt.previous_receipt_hash != previous_hash:
                return False
            if receipt.payload_hash != sha256_hex(canonical_json_bytes(receipt.payload)):
                return False
            if _calculate_receipt_hash(receipt) != receipt.receipt_hash:
                return False
            body_json = _json_object(row["receipt_body_json"])
            if canonical_json_bytes(body_json) != canonical_json_bytes(_receipt_body(receipt)):
                return False
            key = self._resolve_key(receipt.signing_key_id)
            return _verify_receipt_signature(receipt, key)
        except Exception:
            return False

    def _verify_chain_locked(self) -> bool:
        rows = self._connection.execute("SELECT * FROM receipts ORDER BY sequence").fetchall()
        previous_hash: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            if int(row["sequence"]) != expected_sequence:
                return False
            if row["previous_receipt_hash"] != previous_hash:
                return False
            if not self._verify_receipt_row(row, previous_hash):
                return False
            previous_hash = row["receipt_hash"]
        return True

    def verify_chain(self, *, raise_on_error: bool = False) -> bool:
        self._ensure_open()
        with self._lock:
            try:
                valid = self._verify_chain_locked()
            except Exception:
                valid = False
        if not valid and raise_on_error:
            raise ReceiptChainError("audit receipt hash-chain verification failed")
        return valid

    def verify_receipt(
        self,
        receipt: AuditReceipt | str,
        key: bytes | None = None,
    ) -> bool:
        if isinstance(receipt, str):
            receipt = self.get_receipt(receipt)
        try:
            verification_key = (
                _normalise_key(key)
                if key is not None
                else self._resolve_key(receipt.signing_key_id)
            )
            return (
                _calculate_receipt_hash(receipt) == receipt.receipt_hash
                and receipt.payload_hash == sha256_hex(canonical_json_bytes(receipt.payload))
                and _verify_receipt_signature(receipt, verification_key)
            )
        except Exception:
            return False

    @staticmethod
    def _coerce_trace(trace: Trace | Mapping[str, Any]) -> Trace:
        if isinstance(trace, Trace):
            return trace
        if isinstance(trace, Mapping):
            return Trace.model_validate(trace)
        raise StorageError("trace evidence must be a Trace or mapping")

    def register_trace(
        self,
        trace: Trace | Mapping[str, Any],
        job_id: str,
        *,
        actor_id: str = "exhibit-ingestion",
        receipt_id: str | None = None,
        occurred_at: datetime | str | None = None,
        idempotency_key: str | None = None,
    ) -> TraceRegistration:
        trace_model = self._coerce_trace(trace)
        normalised_job_id = _normalise_identifier(job_id, required=True)
        assert normalised_job_id is not None
        canonical_hash = compute_trace_sha256(trace_model)
        if trace_model.trace_sha256 != canonical_hash:
            raise ReceiptConflictError(
                "different trace evidence: declared trace digest does not match canonical spans"
            )

        clean_trace = Trace.model_validate(
            trace_model.model_dump(
                mode="python",
                exclude={
                    "span_count",
                    "event_count",
                    "root_span_ids",
                    "max_depth",
                    "span_digests",
                    "tree_sha256",
                    "trace_sha256",
                },
            )
        )
        trace_json = canonical_json_bytes(clean_trace.model_dump(mode="json")).decode("utf-8")
        effective_receipt_id = receipt_id or (
            f"receipt-trace-{sha256_hex(f'{normalised_job_id}:{clean_trace.trace_id}:{canonical_hash}')[:32]}"
        )

        with self._write_transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM trace_registrations WHERE job_id = ? AND trace_id = ?",
                (normalised_job_id, clean_trace.trace_id),
            ).fetchone()
            if existing_row is not None:
                existing = self._registration_from_row(existing_row)
                if existing.trace_sha256 != canonical_hash:
                    raise ReceiptConflictError(
                        "different trace evidence was supplied for an immutable trace registration"
                    )
                return existing

            registered_at = (
                _normalise_timestamp(occurred_at, required=True)
                if occurred_at is not None
                else self._now()
            )
            assert registered_at is not None
            registration = TraceRegistration(
                job_id=normalised_job_id,
                trace_id=clean_trace.trace_id,
                trace_sha256=canonical_hash,
                source_sha256=clean_trace.source_sha256,
                trace_format=clean_trace.trace_format,
                span_count=clean_trace.span_count or 0,
                event_count=clean_trace.event_count or 0,
                receipt_id=_normalise_identifier(effective_receipt_id, required=True),  # type: ignore[arg-type]
                registered_at=registered_at,
                registration_sha256="0" * 64,
            )
            registration_values = registration.model_dump(mode="python")
            registration_values["registration_sha256"] = sha256_hex(
                {
                    key: value
                    for key, value in _registration_payload(registration_values).items()
                    if key != "registration_sha256"
                }
            )
            registration = TraceRegistration.model_validate(registration_values)

            receipt = self._insert_receipt_locked(
                connection,
                receipt_id=registration.receipt_id,
                event_type="trace.registered",
                actor_type=ReceiptActorType.SERVICE,
                actor_id=actor_id,
                job_id=registration.job_id,
                trace_id=registration.trace_id,
                payload={
                    "trace_id": registration.trace_id,
                    "trace_sha256": registration.trace_sha256,
                    "source_sha256": registration.source_sha256,
                    "trace_format": registration.trace_format.value,
                    "span_count": registration.span_count,
                    "event_count": registration.event_count,
                    "registration_sha256": registration.registration_sha256,
                },
                occurred_at=registered_at,
                idempotency_key=idempotency_key,
            )
            if receipt.receipt_id != registration.receipt_id:
                registration = registration.model_copy(update={"receipt_id": receipt.receipt_id})
                registration_values = registration.model_dump(mode="python")
                registration_values["registration_sha256"] = sha256_hex(
                    {
                        key: value
                        for key, value in _registration_payload(registration_values).items()
                        if key != "registration_sha256"
                    }
                )
                registration = TraceRegistration.model_validate(registration_values)

            connection.execute(
                """
                INSERT INTO trace_registrations (
                    job_id, trace_id, trace_sha256, source_sha256, trace_format,
                    span_count, event_count, trace_json, registration_sha256,
                    receipt_id, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    registration.job_id,
                    registration.trace_id,
                    registration.trace_sha256,
                    registration.source_sha256,
                    registration.trace_format.value,
                    registration.span_count,
                    registration.event_count,
                    trace_json,
                    registration.registration_sha256,
                    registration.receipt_id,
                    _format_timestamp(registration.registered_at),
                ),
            )
            return registration

    @staticmethod
    def _registration_from_row(row: sqlite3.Row) -> TraceRegistration:
        return TraceRegistration.model_validate(
            {
                "job_id": row["job_id"],
                "trace_id": row["trace_id"],
                "trace_sha256": row["trace_sha256"],
                "source_sha256": row["source_sha256"],
                "trace_format": row["trace_format"],
                "span_count": row["span_count"],
                "event_count": row["event_count"],
                "registration_sha256": row["registration_sha256"],
                "receipt_id": row["receipt_id"],
                "registered_at": row["registered_at"],
            }
        )

    def get_trace_registration(self, trace_id: str, job_id: str) -> TraceRegistration:
        self._ensure_open()
        normalised_trace = _normalise_identifier(trace_id, required=True)
        normalised_job = _normalise_identifier(job_id, required=True)
        assert normalised_trace is not None and normalised_job is not None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM trace_registrations WHERE job_id = ? AND trace_id = ?",
                (normalised_job, normalised_trace),
            ).fetchone()
        if row is None:
            raise TraceRegistrationNotFoundError(
                f"trace '{normalised_trace}' is not registered for job '{normalised_job}'"
            )
        return self._registration_from_row(row)

    def get_trace(self, trace_id: str, job_id: str) -> Trace:
        registration = self.get_trace_registration(trace_id, job_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT trace_json FROM trace_registrations "
                "WHERE job_id = ? AND trace_id = ?",
                (registration.job_id, registration.trace_id),
            ).fetchone()
        if row is None:
            raise TraceRegistrationNotFoundError("trace registration disappeared during retrieval")
        return Trace.model_validate(_json_object(row["trace_json"]))

    def list_trace_registrations(self, job_id: str | None = None) -> tuple[TraceRegistration, ...]:
        self._ensure_open()
        query = "SELECT * FROM trace_registrations"
        parameters: tuple[str, ...] = ()
        if job_id is not None:
            query += " WHERE job_id = ?"
            parameters = (_normalise_identifier(job_id, required=True),)  # type: ignore[assignment]
        query += " ORDER BY registered_at, job_id, trace_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._registration_from_row(row) for row in rows)

    @staticmethod
    def _evaluation_from_row(row: sqlite3.Row) -> StoredEvaluation:
        return StoredEvaluation.model_validate(
            {
                "evaluation_id": row["evaluation_id"],
                "job_id": row["job_id"],
                "trace_id": row["trace_id"],
                "trace_sha256": row["trace_sha256"],
                "payload": _json_object(row["payload_json"]),
                "payload_sha256": row["payload_sha256"],
                "receipt_id": row["receipt_id"],
                "stored_at": row["stored_at"],
            }
        )

    def store_evaluation(
        self,
        evaluation: Mapping[str, Any],
        *,
        actor_id: str = "exhibit-evaluator",
        receipt_id: str | None = None,
        occurred_at: datetime | str | None = None,
        idempotency_key: str | None = None,
    ) -> StoredEvaluation:
        if not isinstance(evaluation, Mapping):
            raise StorageError("evaluation evidence must be a mapping")
        payload = canonical_json_bytes(evaluation)
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise StorageError("evaluation exceeds the configured size limit")
        payload_hash = sha256_hex(payload)

        evaluation_id = _normalise_identifier(evaluation.get("evaluation_id"), required=True)
        job_id = _normalise_identifier(evaluation.get("job_id"), required=True)
        trace_id = _normalise_identifier(evaluation.get("trace_id"), required=True)
        assert evaluation_id is not None and job_id is not None and trace_id is not None
        registration = self.get_trace_registration(trace_id, job_id)
        supplied_trace_hash = _normalise_hash(evaluation.get("trace_sha256"))
        if supplied_trace_hash is not None and supplied_trace_hash != registration.trace_sha256:
            raise ReceiptConflictError(
                "different trace evidence was supplied in the evaluation result"
            )

        with self._write_transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM evaluations WHERE job_id = ? AND evaluation_id = ?",
                (job_id, evaluation_id),
            ).fetchone()
            if existing_row is not None:
                existing = self._evaluation_from_row(existing_row)
                if existing.payload_sha256 != payload_hash:
                    raise ReceiptConflictError(
                        "evaluation_id was reused for different evaluation evidence"
                    )
                return existing

            stored_at = (
                _normalise_timestamp(occurred_at, required=True)
                if occurred_at is not None
                else self._now()
            )
            assert stored_at is not None
            effective_receipt_id = receipt_id or (
                f"receipt-evaluation-{sha256_hex(f'{job_id}:{evaluation_id}:{payload_hash}')[:32]}"
            )
            receipt = self._insert_receipt_locked(
                connection,
                receipt_id=effective_receipt_id,
                event_type="evaluation.stored",
                actor_type=ReceiptActorType.EVALUATOR,
                actor_id=actor_id,
                job_id=job_id,
                trace_id=trace_id,
                payload={
                    "evaluation_id": evaluation_id,
                    "job_id": job_id,
                    "trace_id": trace_id,
                    "trace_sha256": registration.trace_sha256,
                    "evaluation_sha256": payload_hash,
                },
                occurred_at=stored_at,
                idempotency_key=idempotency_key,
            )
            record = StoredEvaluation(
                evaluation_id=evaluation_id,
                job_id=job_id,
                trace_id=trace_id,
                trace_sha256=registration.trace_sha256,
                payload=_json_object(payload.decode("utf-8")),
                payload_sha256=payload_hash,
                receipt_id=receipt.receipt_id,
                stored_at=stored_at,
            )
            connection.execute(
                """
                INSERT INTO evaluations (
                    evaluation_id, job_id, trace_id, trace_sha256,
                    payload_json, payload_sha256, receipt_id, stored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.evaluation_id,
                    record.job_id,
                    record.trace_id,
                    record.trace_sha256,
                    canonical_json_bytes(record.payload).decode("utf-8"),
                    record.payload_sha256,
                    record.receipt_id,
                    _format_timestamp(record.stored_at),
                ),
            )
            return record

    def get_evaluation(
        self,
        evaluation_id: str,
        job_id: str | None = None,
    ) -> StoredEvaluation:
        self._ensure_open()
        normalised_evaluation = _normalise_identifier(evaluation_id, required=True)
        normalised_job = _normalise_identifier(job_id) if job_id is not None else None
        assert normalised_evaluation is not None
        query = "SELECT * FROM evaluations WHERE evaluation_id = ?"
        parameters: list[str] = [normalised_evaluation]
        if normalised_job is not None:
            query += " AND job_id = ?"
            parameters.append(normalised_job)
        query += " ORDER BY job_id LIMIT 1"
        with self._lock:
            row = self._connection.execute(query, parameters).fetchone()
        if row is None:
            scope = f" for job '{normalised_job}'" if normalised_job else ""
            raise EvaluationNotFoundError(
                f"evaluation '{normalised_evaluation}' does not exist{scope}"
            )
        return self._evaluation_from_row(row)

    def list_evaluations(
        self,
        *,
        job_id: str | None = None,
        trace_id: str | None = None,
    ) -> tuple[StoredEvaluation, ...]:
        self._ensure_open()
        clauses: list[str] = []
        parameters: list[str] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            parameters.append(_normalise_identifier(job_id, required=True))  # type: ignore[arg-type]
        if trace_id is not None:
            clauses.append("trace_id = ?")
            parameters.append(_normalise_identifier(trace_id, required=True))  # type: ignore[arg-type]
        query = "SELECT * FROM evaluations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY stored_at, job_id, evaluation_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._evaluation_from_row(row) for row in rows)

    def record_human_review(
        self,
        review: HumanReviewSignoff | Mapping[str, Any],
        *,
        occurred_at: datetime | str | None = None,
        idempotency_key: str | None = None,
    ) -> HumanReviewRecord:
        try:
            signoff = (
                review
                if isinstance(review, HumanReviewSignoff)
                else HumanReviewSignoff.model_validate(review)
            )
            signoff = HumanReviewSignoff.model_validate(signoff.model_dump(mode="python"))
        except Exception as error:
            if isinstance(error, InvalidHumanReviewError):
                raise
            raise InvalidHumanReviewError(str(error)) from error
        if signoff.actor != "human":
            raise InvalidHumanReviewError("actor='human' is required for a review sign-off")
        try:
            key = self._resolve_key(signoff.signing_key_id)
        except ReceiptChainError as error:
            raise InvalidHumanReviewError(
                f"no trusted key is available for review key ID '{signoff.signing_key_id}'"
            ) from error
        if not verify_human_review_bytes(signoff.signing_payload, key, signoff.signature):
            raise InvalidHumanReviewError("human-review signature verification failed")

        signoff_json = canonical_json_bytes(signoff.model_dump(mode="json")).decode("utf-8")
        signoff_hash = sha256_hex(signoff_json)

        with self._write_transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM human_reviews WHERE job_id = ? AND signoff_id = ?",
                (signoff.job_id, signoff.signoff_id),
            ).fetchone()
            if existing_row is not None:
                existing = self._review_from_row(existing_row)
                if existing.signoff_sha256 != signoff_hash:
                    raise ReceiptConflictError(
                        "signoff_id was reused for different human-review evidence"
                    )
                return existing

            recorded_at = (
                _normalise_timestamp(occurred_at, required=True)
                if occurred_at is not None
                else self._now()
            )
            assert recorded_at is not None
            receipt = self._insert_receipt_locked(
                connection,
                receipt_id=signoff.receipt_id or f"receipt-review-{signoff_hash[:32]}",
                event_type="human.review.signed",
                actor_type=ReceiptActorType.HUMAN,
                actor_id=signoff.reviewer_id,
                job_id=signoff.job_id,
                trace_id=signoff.trace_id,
                payload={
                    "signoff": signoff.model_dump(mode="json"),
                    "signoff_sha256": signoff_hash,
                },
                occurred_at=recorded_at,
                idempotency_key=idempotency_key,
            )
            record = HumanReviewRecord(
                signoff_id=signoff.signoff_id,
                job_id=signoff.job_id,
                trace_id=signoff.trace_id,
                trace_sha256=signoff.trace_sha256,
                signoff=signoff,
                signoff_sha256=signoff_hash,
                receipt_id=receipt.receipt_id,
                recorded_at=recorded_at,
            )
            connection.execute(
                """
                INSERT INTO human_reviews (
                    signoff_id, job_id, trace_id, trace_sha256,
                    signoff_json, signoff_sha256, receipt_id, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.signoff_id,
                    record.job_id,
                    record.trace_id,
                    record.trace_sha256,
                    signoff_json,
                    record.signoff_sha256,
                    record.receipt_id,
                    _format_timestamp(record.recorded_at),
                ),
            )
            return record

    def store_human_review(
        self,
        review: HumanReviewSignoff | Mapping[str, Any],
        **kwargs: Any,
    ) -> HumanReviewRecord:
        return self.record_human_review(review, **kwargs)

    @staticmethod
    def _review_from_row(row: sqlite3.Row) -> HumanReviewRecord:
        return HumanReviewRecord.model_validate(
            {
                "signoff_id": row["signoff_id"],
                "job_id": row["job_id"],
                "trace_id": row["trace_id"],
                "trace_sha256": row["trace_sha256"],
                "signoff": _json_object(row["signoff_json"]),
                "signoff_sha256": row["signoff_sha256"],
                "receipt_id": row["receipt_id"],
                "recorded_at": row["recorded_at"],
            }
        )

    def get_human_review_record(
        self,
        signoff_id: str,
        job_id: str,
    ) -> HumanReviewRecord:
        self._ensure_open()
        normalised_signoff = _normalise_identifier(signoff_id, required=True)
        normalised_job = _normalise_identifier(job_id, required=True)
        assert normalised_signoff is not None and normalised_job is not None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM human_reviews WHERE job_id = ? AND signoff_id = ?",
                (normalised_job, normalised_signoff),
            ).fetchone()
        if row is None:
            raise HumanReviewNotFoundError(
                f"human review '{normalised_signoff}' does not exist for job '{normalised_job}'"
            )
        return self._review_from_row(row)

    def get_human_review(self, signoff_id: str, job_id: str) -> HumanReviewSignoff:
        return self.get_human_review_record(signoff_id, job_id).signoff

    def list_human_reviews(
        self,
        *,
        job_id: str | None = None,
        trace_id: str | None = None,
    ) -> tuple[HumanReviewRecord, ...]:
        self._ensure_open()
        clauses: list[str] = []
        parameters: list[str] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            parameters.append(_normalise_identifier(job_id, required=True))  # type: ignore[arg-type]
        if trace_id is not None:
            clauses.append("trace_id = ?")
            parameters.append(_normalise_identifier(trace_id, required=True))  # type: ignore[arg-type]
        query = "SELECT * FROM human_reviews"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY recorded_at, job_id, signoff_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._review_from_row(row) for row in rows)

    def _verify_record_integrity_locked(self) -> bool:
        receipt_ids = {
            row["receipt_id"]
            for row in self._connection.execute("SELECT receipt_id FROM receipts").fetchall()
        }

        for row in self._connection.execute("SELECT * FROM trace_registrations").fetchall():
            try:
                registration = self._registration_from_row(row)
                if registration.receipt_id not in receipt_ids:
                    return False
                unsigned = {
                    key: value
                    for key, value in _registration_payload(registration).items()
                    if key != "registration_sha256"
                }
                if registration.registration_sha256 != sha256_hex(unsigned):
                    return False
                trace = Trace.model_validate(_json_object(row["trace_json"]))
                if (
                    trace.trace_id != registration.trace_id
                    or trace.trace_sha256 != registration.trace_sha256
                    or compute_trace_sha256(trace) != registration.trace_sha256
                ):
                    return False
            except Exception:
                return False

        for row in self._connection.execute("SELECT * FROM evaluations").fetchall():
            try:
                evaluation = self._evaluation_from_row(row)
                if evaluation.receipt_id not in receipt_ids:
                    return False
                if evaluation.payload_sha256 != sha256_hex(
                    canonical_json_bytes(evaluation.payload)
                ):
                    return False
                registration = self.get_trace_registration(
                    evaluation.trace_id,
                    evaluation.job_id,
                )
                if registration.trace_sha256 != evaluation.trace_sha256:
                    return False
            except Exception:
                return False

        for row in self._connection.execute("SELECT * FROM human_reviews").fetchall():
            try:
                record = self._review_from_row(row)
                if record.receipt_id not in receipt_ids:
                    return False
                serialised = canonical_json_bytes(record.signoff.model_dump(mode="json"))
                if record.signoff_sha256 != sha256_hex(serialised):
                    return False
                key = self._resolve_key(record.signoff.signing_key_id)
                if not verify_human_review_bytes(
                    record.signoff.signing_payload,
                    key,
                    record.signoff.signature,
                ):
                    return False
            except Exception:
                return False
        return True

    def verify_database_integrity(self, *, raise_on_error: bool = False) -> bool:
        self._ensure_open()
        valid = False
        with self._lock:
            try:
                quick_check = self._connection.execute("PRAGMA quick_check").fetchall()
                valid = (
                    len(quick_check) == 1
                    and quick_check[0][0] == "ok"
                    and self._verify_chain_locked()
                    and self._verify_record_integrity_locked()
                )
            except Exception:
                valid = False
        if not valid and raise_on_error:
            raise ReceiptChainError("receipt database integrity verification failed")
        return valid

    def assert_database_integrity(self) -> None:
        self.verify_database_integrity(raise_on_error=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._connection.close()
            finally:
                self._closed = True

    def __enter__(self) -> ReceiptsStore:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False