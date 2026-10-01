from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


UTC = timezone.utc
CONSTITUTIONAL_RULE = "Not a conformity assessment. Counsel classifies."

_IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f]+$")
_EVENT_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")
_HUMAN_REVIEW_CONTEXT = b"exhibit/human-review-signoff/v1\x00"


class _FrozenDict(dict[str, Any]):
    """A recursively immutable mapping used by evidence models."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if values:
            dict.update(self, values)

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("evidence mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        memo[id(self)] = self
        return self

    def __copy__(self) -> _FrozenDict:
        return self


def _freeze_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _freeze_value(value.model_dump(mode="python", by_alias=True))
    if isinstance(value, Mapping):
        return _FrozenDict({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        encoded = [_freeze_value(item) for item in value]
        return tuple(
            sorted(
                encoded,
                key=lambda item: json.dumps(
                    _canonical_value(item),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python", by_alias=True))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal cannot be canonicalized")
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_canonical_value(item) for item in value]
        return sorted(encoded, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floating point value cannot be canonicalized")
    return value


def _json_default(value: Any) -> Any:
    canonical = _canonical_value(value)
    if canonical is not value:
        return canonical
    raise TypeError(f"value of type {type(value).__name__} is not canonically serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON suitable for cryptographic hashing."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")


def sha256_hex(value: bytes | bytearray | memoryview | str | Any) -> str:
    """Hash bytes or strings directly and all other values through canonical JSON."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        digest_input = bytes(value)
    elif isinstance(value, str):
        digest_input = value.encode("utf-8")
    else:
        digest_input = canonical_json_bytes(value)
    return hashlib.sha256(digest_input).hexdigest()


def _normalise_identifier(
    value: Any,
    *,
    width: int | None = None,
    required: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise ValueError("identifier is required")
        return None
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid identifiers")
    if isinstance(value, UUID):
        text = value.hex
    elif isinstance(value, int):
        if value < 0:
            raise ValueError("numeric identifiers cannot be negative")
        text = format(value, "x")
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError("identifier must be a string, UUID, or non-negative integer")

    if not text:
        if required:
            raise ValueError("identifier is required")
        return None
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError("identifier contains a forbidden control character")
    if text.startswith("0x"):
        text = text[2:]

    hexadecimal = bool(text) and all(character in "0123456789abcdefABCDEF" for character in text)
    text = text.lower()
    if hexadecimal:
        if width is not None and len(text) > width:
            raise ValueError(f"identifier exceeds {width} hexadecimal characters")
        if width is not None:
            text = text.zfill(width)
    elif len(text) > 256:
        raise ValueError("identifier exceeds 256 characters")
    return text


def _normalise_hash(value: Any, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError("SHA-256 digest is required")
        return None
    if not isinstance(value, str):
        raise ValueError("SHA-256 digest must be a hexadecimal string")
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("SHA-256 digest must contain exactly 64 hexadecimal characters")
    return digest


def _normalise_timestamp(value: Any, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise ValueError("timestamp is required")
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError("timestamp must be an RFC 3339 or ISO 8601 datetime") from error
    if not isinstance(value, datetime):
        raise ValueError("timestamp must be a datetime or RFC 3339 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


def _normalise_event_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("event type must be a string")
    event_type = value.strip().casefold()
    if not _EVENT_TYPE_RE.fullmatch(event_type):
        raise ValueError("event type must use lowercase letters, digits, '.', ':', '_' or '-'")
    return event_type


def _coerce_key(value: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("HMAC keys must be supplied as bytes")
    key = bytes(value)
    if len(key) < 32:
        raise ValueError("HMAC keys must contain at least 32 bytes")
    return key


def sign_human_review_bytes(payload: bytes | bytearray | memoryview, key: bytes) -> str:
    """Create a domain-separated HMAC-SHA256 signature for review evidence."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("human-review signing payload must be bytes")
    signing_key = _coerce_key(key)
    digest = hmac.new(
        signing_key,
        _HUMAN_REVIEW_CONTEXT + bytes(payload),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


def verify_human_review_bytes(
    payload: bytes | bytearray | memoryview,
    key: bytes,
    signature: str,
) -> bool:
    """Verify a human-review signature without leaking comparison timing."""

    if not isinstance(payload, (bytes, bytearray | memoryview)):
        raise TypeError("human-review signing payload must be bytes")
    if not isinstance(signature, str) or not _BASE64_RE.fullmatch(signature):
        return False
    try:
        supplied = base64.b64decode(signature.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError):
        return False
    if len(supplied) != hashlib.sha256().digest_size:
        return False
    expected = base64.urlsafe_b64decode(sign_human_review_bytes(payload, key).encode("ascii"))
    return hmac.compare_digest(supplied, expected)


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
    )


class SpanKind(str, Enum):
    UNSPECIFIED = "unspecified"
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    GUARDRAIL = "guardrail"
    EVALUATOR = "evaluator"

    @classmethod
    def _missing_(cls, value: object) -> SpanKind | None:
        if not isinstance(value, str):
            return None
        token = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
        aliases = {
            "internal": cls.INTERNAL,
            "server": cls.SERVER,
            "client": cls.CLIENT,
            "producer": cls.PRODUCER,
            "consumer": cls.CONSUMER,
            "agent": cls.AGENT,
            "llm": cls.LLM,
            "large_language_model": cls.LLM,
            "model": cls.LLM,
            "tool": cls.TOOL,
            "chain": cls.CHAIN,
            "workflow": cls.CHAIN,
            "retriever": cls.RETRIEVER,
            "retrieval": cls.RETRIEVER,
            "embedding": cls.EMBEDDING,
            "embeddings": cls.EMBEDDING,
            "reranker": cls.RERANKER,
            "rerank": cls.RERANKER,
            "guardrail": cls.GUARDRAIL,
            "evaluator": cls.EVALUATOR,
            "evaluation": cls.EVALUATOR,
            "unspecified": cls.UNSPECIFIED,
        }
        return aliases.get(token)


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"

    @classmethod
    def _missing_(cls, value: object) -> SpanStatus | None:
        if not isinstance(value, str):
            return None
        token = value.strip().casefold()
        aliases = {
            "ok": cls.OK,
            "success": cls.OK,
            "unset": cls.UNSET,
            "": cls.UNSET,
            "error": cls.ERROR,
            "failure": cls.ERROR,
            "failed": cls.ERROR,
        }
        return aliases.get(token)


class TraceFormat(str, Enum):
    OPENINFERENCE = "openinference"
    OTLP = "otlp"
    JSONL = "jsonl"
    AGGREGATED = "aggregated"
    MIXED = "mixed"

    @classmethod
    def _missing_(cls, value: object) -> TraceFormat | None:
        if not isinstance(value, str):
            return None
        token = re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")
        aliases = {
            "open-inference": cls.OPENINFERENCE,
            "openinference-jsonl": cls.OPENINFERENCE,
            "json-lines": cls.JSONL,
            "jsonl": cls.JSONL,
            "opentelemetry": cls.OTLP,
            "otel": cls.OTLP,
            "aggregated": cls.AGGREGATED,
            "mixed": cls.MIXED,
        }
        return aliases.get(token)


class EvidenceDigest(_EvidenceModel):
    evidence_id: str
    kind: str = "evidence"
    algorithm: str = "SHA-256"
    value: str = Field(
        validation_alias=AliasChoices("value", "digest", "hash", "sha256")
    )
    source_span_ids: tuple[str, ...] = ()
    verified_at: datetime | None = None

    @field_validator("evidence_id", "kind", mode="before")
    @classmethod
    def validate_identifier(cls, value: Any) -> str:
        normalised = _normalise_identifier(value, required=True)
        assert normalised is not None
        return normalised

    @field_validator("algorithm", mode="before")
    @classmethod
    def normalise_algorithm(cls, value: Any) -> str:
        algorithm = str(value).strip().upper().replace("_", "-")
        if algorithm not in {"SHA-256", "SHA256"}:
            raise ValueError("only SHA-256 evidence digests are supported")
        return "SHA-256"

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: Any) -> str:
        digest = _normalise_hash(value, required=True)
        assert digest is not None
        return digest

    @field_validator("source_span_ids", mode="before")
    @classmethod
    def validate_sources(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            raise ValueError("source_span_ids must be a sequence")
        result: list[str] = []
        for item in value:
            identifier = _normalise_identifier(item, required=True)
            assert identifier is not None
            result.append(identifier)
        return tuple(result)

    @field_validator("verified_at", mode="before")
    @classmethod
    def validate_verified_at(cls, value: Any) -> datetime | None:
        return _normalise_timestamp(value)


class EvidenceReadiness(_EvidenceModel):
    status: str = "not-evidenced"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = "No engineering readiness claim has been supplied."
    evidence: tuple[EvidenceDigest, ...] = ()
    gaps: tuple[str, ...] = ()

    @field_validator("status", mode="before")
    @classmethod
    def normalise_status(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "evidenced" if value else "not-evidenced"
        token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().casefold()).strip("-")
        aliases = {
            "ready": "evidenced",
            "complete": "evidenced",
            "evidenced": "evidenced",
            "available": "evidenced",
            "partial": "partial",
            "incomplete": "partial",
            "not-ready": "not-evidenced",
            "not-evidenced": "not-evidenced",
            "missing": "not-evidenced",
            "absent": "not-evidenced",
        }
        if token not in aliases:
            raise ValueError("readiness status must be evidenced, partial, or not-evidenced")
        return aliases[token]

    @field_validator("gaps", mode="before")
    @classmethod
    def validate_gaps(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("gaps must be a sequence of strings")
        return tuple(str(item).strip() for item in value if str(item).strip())


class SpanEvent(_EvidenceModel):
    name: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(
        validation_alias=AliasChoices("timestamp", "time", "time_unix_nano", "ts")
    )
    attributes: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalised = value.strip()
        if not _IDENTIFIER_RE.fullmatch(normalised):
            raise ValueError("event name contains a forbidden control character")
        return normalised

    @field_validator("timestamp", mode="before")
    @classmethod
    def normalise_timestamp(cls, value: Any) -> datetime:
        timestamp = _normalise_timestamp(value, required=True)
        assert timestamp is not None
        return timestamp

    @field_validator("attributes", mode="before")
    @classmethod
    def normalise_attributes(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("event attributes must be a mapping")
        return value

    @model_validator(mode="after")
    def freeze_evidence(self) -> SpanEvent:
        object.__setattr__(self, "attributes", _freeze_value(self.attributes))
        return self

    @property
    def sha256(self) -> str:
        return sha256_hex(self.model_dump(mode="python"))


class SpanLink(_EvidenceModel):
    trace_id: str | None = None
    span_id: str = Field(
        validation_alias=AliasChoices("span_id", "spanId", "spanID", "linked_span_id")
    )
    trace_state: Mapping[str, Any] | None = None
    attributes: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str | None:
        return _normalise_identifier(value)

    @field_validator("span_id", mode="before")
    @classmethod
    def normalise_span_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, required=True)
        assert identifier is not None
        return identifier

    @field_validator("attributes", "trace_state", mode="before")
    @classmethod
    def normalise_mapping(cls, value: Any) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("link metadata must be a mapping")
        return value

    @model_validator(mode="after")
    def freeze_evidence(self) -> SpanLink:
        object.__setattr__(self, "attributes", _freeze_value(self.attributes))
        if self.trace_state is not None:
            object.__setattr__(self, "trace_state", _freeze_value(self.trace_state))
        return self


class Span(_EvidenceModel):
    trace_id: str = Field(validation_alias=AliasChoices("trace_id", "traceId", "traceID"))
    span_id: str = Field(validation_alias=AliasChoices("span_id", "spanId", "spanID", "id"))
    parent_span_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "parent_span_id",
            "parentSpanId",
            "parent_span_ID",
            "parent_id",
            "parentId",
        ),
    )
    name: str = Field(
        min_length=1,
        max_length=512,
        validation_alias=AliasChoices("name", "span_name", "spanName", "operation"),
    )
    kind: SpanKind = Field(default=SpanKind.INTERNAL)
    start_time: datetime = Field(
        validation_alias=AliasChoices("start_time", "startTime", "start", "timestamp")
    )
    end_time: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("end_time", "endTime", "end"),
    )
    status: SpanStatus = Field(default=SpanStatus.UNSET)
    status_message: str | None = Field(default=None, max_length=2_000)
    attributes: Mapping[str, Any] = Field(default_factory=dict)
    events: tuple[SpanEvent, ...] = ()
    links: tuple[SpanLink, ...] = ()
    resource_attributes: Mapping[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("resource_attributes", "resourceAttributes"),
    )
    dropped_attributes_count: int = Field(default=0, ge=0)
    dropped_events_count: int = Field(default=0, ge=0)
    dropped_links_count: int = Field(default=0, ge=0)

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("span_id", mode="before")
    @classmethod
    def normalise_span_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=16, required=True)
        assert identifier is not None
        return identifier

    @field_validator("parent_span_id", mode="before")
    @classmethod
    def normalise_parent_span_id(cls, value: Any) -> str | None:
        return _normalise_identifier(value, width=16)

    @field_validator("name", mode="before")
    @classmethod
    def normalise_name(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("span name must be a string")
        name = value.strip()
        if not name or not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError("span name is empty or contains a control character")
        return name

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def normalise_times(cls, value: Any) -> datetime | None:
        return _normalise_timestamp(value)

    @field_validator("status_message", mode="before")
    @classmethod
    def normalise_status_message(cls, value: Any) -> str | None:
        if value is None:
            return None
        message = str(value).strip()
        return message or None

    @field_validator("attributes", "resource_attributes", mode="before")
    @classmethod
    def normalise_attributes(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("span attributes must be mappings")
        return value

    @field_validator("events", "links", mode="before")
    @classmethod
    def normalise_sequence(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError("events and links must be sequences")
        return tuple(value)

    @model_validator(mode="after")
    def validate_span(self) -> Span:
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time cannot precede start_time")
        object.__setattr__(self, "attributes", _freeze_value(self.attributes))
        object.__setattr__(
            self,
            "resource_attributes",
            _freeze_value(self.resource_attributes),
        )
        return self

    @property
    def sha256(self) -> str:
        return sha256_hex(self.model_dump(mode="python"))

    @property
    def evidence_sha256(self) -> str:
        return self.sha256

    @property
    def event_count(self) -> int:
        return len(self.events)


_TRACE_DERIVED_FIELDS = {
    "span_count",
    "event_count",
    "root_span_ids",
    "max_depth",
    "span_digests",
    "tree_sha256",
    "trace_sha256",
}


def _trace_unsigned_payload(trace: Trace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, Trace):
        payload = trace.model_dump(mode="python")
    elif isinstance(trace, Mapping):
        payload = dict(trace)
    else:
        raise TypeError("trace evidence must be a Trace or mapping")
    for field in _TRACE_DERIVED_FIELDS:
        payload.pop(field, None)
    return payload


def _trace_derived_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_spans = payload.get("spans", ())
    if isinstance(raw_spans, (str, bytes, Mapping)):
        raise ValueError("trace spans must be a sequence")
    spans = tuple(
        span if isinstance(span, Span) else Span.model_validate(span)
        for span in raw_spans
    )
    span_ids = {span.span_id for span in spans}
    parent_by_span: dict[str, str | None] = {}
    for span in spans:
        if span.parent_span_id is not None and span.parent_span_id not in span_ids:
            raise ValueError(
                f"span '{span.span_id}' references missing parent "
                f"'{span.parent_span_id}'"
            )
        parent_by_span.setdefault(span.span_id, span.parent_span_id)

    def depth(span_id: str, path: frozenset[str] = frozenset()) -> int:
        if span_id in path:
            raise ValueError("trace span tree contains a parent cycle")
        parent = parent_by_span.get(span_id)
        if parent is None:
            return 0
        return 1 + depth(parent, path | {span_id})

    sorted_spans = sorted(
        spans,
        key=lambda span: canonical_json_bytes(span.model_dump(mode="python")),
    )
    tree_sha256 = sha256_hex([span.model_dump(mode="python") for span in sorted_spans])
    digests = tuple(
        EvidenceDigest(
            evidence_id=f"span:{span.span_id}:{index}",
            kind="span",
            value=sha256_hex(span.model_dump(mode="python")),
            source_span_ids=(span.span_id,),
        )
        for index, span in enumerate(sorted_spans)
    )
    return {
        "span_count": len(spans),
        "event_count": sum(len(span.events) for span in spans),
        "root_span_ids": tuple(
            sorted({span.span_id for span in spans if span.parent_span_id is None})
        ),
        "max_depth": max((depth(span.span_id) for span in spans), default=0),
        "span_digests": digests,
        "tree_sha256": tree_sha256,
    }


def compute_trace_sha256(trace: Trace | Mapping[str, Any]) -> str:
    """Compute the canonical SHA-256 digest of trace evidence.

    Cached digest and count fields are deliberately excluded. This function is
    used by persistence code so a ``model_copy(update=...)`` cannot preserve a
    stale digest after span evidence has changed.
    """

    unsigned = _trace_unsigned_payload(trace)
    derived = _trace_derived_values(unsigned)
    return sha256_hex({"trace": unsigned, "tree_sha256": derived["tree_sha256"]})


class Trace(_EvidenceModel):
    trace_id: str = Field(validation_alias=AliasChoices("trace_id", "traceId", "traceID"))
    source_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_name", "sourceName", "source"),
    )
    source_sha256: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_sha256", "sourceSha256", "source_hash"),
    )
    trace_format: TraceFormat = TraceFormat.OPENINFERENCE
    spans: tuple[Span, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("spans", "span_records", "records"),
    )
    ingested_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("ingested_at", "ingestedAt", "created_at"),
    )
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    span_count: int | None = Field(default=None, ge=0)
    event_count: int | None = Field(default=None, ge=0)
    root_span_ids: tuple[str, ...] | None = None
    max_depth: int | None = Field(default=None, ge=0)
    span_digests: tuple[EvidenceDigest, ...] | None = None
    tree_sha256: str | None = None
    trace_sha256: str | None = Field(
        default=None,
        validation_alias=AliasChoices("trace_sha256", "traceSha256", "trace_hash"),
    )

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("source_sha256", mode="before")
    @classmethod
    def normalise_source_hash(cls, value: Any) -> str | None:
        return _normalise_hash(value)

    @field_validator("source_name", mode="before")
    @classmethod
    def normalise_source_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        name = str(value).strip()
        return name or None

    @field_validator("spans", mode="before")
    @classmethod
    def normalise_spans(cls, value: Any) -> tuple[Any, ...]:
        if value is None or isinstance(value, (str, bytes, Mapping)):
            raise ValueError("spans must be a non-empty sequence")
        return tuple(value)

    @field_validator("root_span_ids", mode="before")
    @classmethod
    def normalise_roots(cls, value: Any) -> tuple[str, ...] | None:
        if value is None:
            return None
        return tuple(_normalise_identifier(item, required=True) for item in value)  # type: ignore[misc]

    @field_validator("tree_sha256", "trace_sha256", mode="before")
    @classmethod
    def normalise_digest(cls, value: Any) -> str | None:
        return _normalise_hash(value)

    @field_validator("ingested_at", mode="before")
    @classmethod
    def normalise_ingested_at(cls, value: Any) -> datetime | None:
        return _normalise_timestamp(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def normalise_metadata(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("trace metadata must be a mapping")
        return value

    @model_validator(mode="after")
    def validate_and_derive_trace(self) -> Trace:
        for span in self.spans:
            if span.trace_id != self.trace_id:
                raise ValueError(
                    f"span '{span.span_id}' belongs to trace '{span.trace_id}', "
                    f"not '{self.trace_id}'"
                )

        unsigned = _trace_unsigned_payload(self)
        derived = _trace_derived_values(unsigned)
        expected_trace_hash = compute_trace_sha256(unsigned)

        declared = {
            "span_count": self.span_count,
            "event_count": self.event_count,
            "root_span_ids": self.root_span_ids,
            "max_depth": self.max_depth,
            "span_digests": self.span_digests,
            "tree_sha256": self.tree_sha256,
            "trace_sha256": self.trace_sha256,
        }
        for name, expected in derived.items():
            supplied = declared[name]
            if supplied is not None and supplied != expected:
                raise ValueError(f"{name} does not match canonical trace evidence")
        if self.trace_sha256 is not None and self.trace_sha256 != expected_trace_hash:
            raise ValueError("trace_sha256 does not match canonical trace evidence")

        object.__setattr__(self, "metadata", _freeze_value(self.metadata))
        for name, value in derived.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "trace_sha256", expected_trace_hash)
        return self

    @property
    def root_span_id(self) -> str | None:
        return self.root_span_ids[0] if self.root_span_ids else None

    @property
    def evidence_sha256(self) -> str:
        return self.trace_sha256 or compute_trace_sha256(self)

    @property
    def span_ids(self) -> tuple[str, ...]:
        return tuple(span.span_id for span in self.spans)


class EvaluationScore(_EvidenceModel):
    name: str = Field(min_length=1, max_length=128)
    passed: bool = Field(strict=True)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(
        default="No criterion rationale was supplied.",
        validation_alias=AliasChoices("reason", "rationale"),
        max_length=4_000,
    )

    @field_validator("name", mode="before")
    @classmethod
    def normalise_name(cls, value: Any) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("evaluation criterion name is required")
        return name

    @field_validator("reason", mode="before")
    @classmethod
    def normalise_reason(cls, value: Any) -> str:
        reason = str(value).strip()
        return reason or "No criterion rationale was supplied."


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    APPROVED_WITH_LIMITATIONS = "approved-with-limitations"
    REJECTED = "rejected"

    @classmethod
    def _missing_(cls, value: object) -> ReviewDecision | None:
        if not isinstance(value, str):
            return None
        token = re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")
        aliases = {
            "approved": cls.APPROVED,
            "approve": cls.APPROVED,
            "accepted": cls.APPROVED,
            "approved-with-limitations": cls.APPROVED_WITH_LIMITATIONS,
            "conditionally-approved": cls.APPROVED_WITH_LIMITATIONS,
            "rejected": cls.REJECTED,
            "reject": cls.REJECTED,
            "denied": cls.REJECTED,
        }
        return aliases.get(token)


class HumanReviewSignoff(_EvidenceModel):
    signoff_id: str
    job_id: str
    trace_id: str
    trace_sha256: str
    actor: str
    reviewer_id: str
    reviewer_name: str
    decision: ReviewDecision
    attestation: str = Field(min_length=1, max_length=8_000)
    signed_at: datetime
    scope: tuple[str, ...] = Field(min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    receipt_id: str | None = None
    signing_key_id: str = Field(default="primary")
    signature: str

    @field_validator("signoff_id", "job_id", "reviewer_id", mode="before")
    @classmethod
    def normalise_required_identifiers(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_sha256", mode="before")
    @classmethod
    def normalise_trace_hash(cls, value: Any) -> str:
        digest = _normalise_hash(value, required=True)
        assert digest is not None
        return digest

    @field_validator("actor", mode="before")
    @classmethod
    def normalise_actor(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("actor='human' is required for a human-review sign-off")
        actor = value.strip().casefold().replace("_", "-")
        aliases = {
            "human-reviewer": "human",
            "reviewer": "human",
            "operator": "human",
        }
        return aliases.get(actor, actor)

    @field_validator("reviewer_name", "attestation", mode="before")
    @classmethod
    def normalise_required_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text or not _IDENTIFIER_RE.fullmatch(text):
            raise ValueError("reviewer name and attestation must be non-empty text")
        return text

    @field_validator("signed_at", mode="before")
    @classmethod
    def normalise_signed_at(cls, value: Any) -> datetime:
        timestamp = _normalise_timestamp(value, required=True)
        assert timestamp is not None
        return timestamp

    @field_validator("scope", mode="before")
    @classmethod
    def normalise_scope(cls, value: Any) -> tuple[str, ...]:
        if value is None or isinstance(value, (str, bytes, Mapping)):
            raise ValueError("review scope must be a non-empty sequence")
        scope = tuple(str(item).strip() for item in value if str(item).strip())
        if not scope:
            raise ValueError("review scope must contain at least one item")
        return scope

    @field_validator("metadata", mode="before")
    @classmethod
    def normalise_metadata(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("review metadata must be a mapping")
        return value

    @field_validator("receipt_id", mode="before")
    @classmethod
    def normalise_receipt_id(cls, value: Any) -> str | None:
        return _normalise_identifier(value)

    @field_validator("signing_key_id", mode="before")
    @classmethod
    def normalise_signing_key_id(cls, value: Any) -> str:
        key_id = str(value).strip()
        if not _KEY_ID_RE.fullmatch(key_id):
            raise ValueError("signing key ID is invalid")
        return key_id

    @field_validator("signature", mode="before")
    @classmethod
    def validate_signature_encoding(cls, value: Any) -> str:
        if not isinstance(value, str) or not _BASE64_RE.fullmatch(value):
            raise ValueError("human-review signature must be base64url text")
        try:
            decoded = base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as error:
            raise ValueError("human-review signature must be valid base64url") from error
        if len(decoded) != hashlib.sha256().digest_size:
            raise ValueError("human-review signature must be a SHA-256 HMAC")
        return value

    @model_validator(mode="after")
    def enforce_human_actor(self) -> HumanReviewSignoff:
        if self.actor != "human":
            raise ValueError("actor='human' is required for a human-review sign-off")
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))
        return self

    def _signing_mapping(self) -> dict[str, Any]:
        return {
            "signoff_id": self.signoff_id,
            "job_id": self.job_id,
            "trace_id": self.trace_id,
            "trace_sha256": self.trace_sha256,
            "actor": self.actor,
            "reviewer_id": self.reviewer_id,
            "reviewer_name": self.reviewer_name,
            "decision": self.decision,
            "attestation": self.attestation,
            "signed_at": self.signed_at,
            "scope": self.scope,
            "metadata": self.metadata,
            "receipt_id": self.receipt_id,
            "signing_key_id": self.signing_key_id,
        }

    @property
    def signing_payload(self) -> bytes:
        return canonical_json_bytes(self._signing_mapping())

    def verify_signature(self, key: bytes) -> bool:
        return verify_human_review_bytes(self.signing_payload, key, self.signature)

    @property
    def approved(self) -> bool:
        return self.decision in {
            ReviewDecision.APPROVED,
            ReviewDecision.APPROVED_WITH_LIMITATIONS,
        }


class SecuritySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def _missing_(cls, value: object) -> SecuritySeverity | None:
        if not isinstance(value, str):
            return None
        token = value.strip().casefold()
        aliases = {
            "informational": cls.INFO,
            "warning": cls.MEDIUM,
            "severe": cls.CRITICAL,
        }
        return aliases.get(token, token)


class SecurityThreat(_EvidenceModel):
    threat_id: str
    category: str
    severity: SecuritySeverity
    description: str
    source_span_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    status: str = "open"
    mitigation: str | None = None
    article_reference: str = "Article 15"

    @field_validator("threat_id", "category", mode="before")
    @classmethod
    def normalise_identifier(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, required=True)
        assert identifier is not None
        return identifier

    @field_validator("description", "status", "mitigation", mode="before")
    @classmethod
    def normalise_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("source_span_ids", mode="before")
    @classmethod
    def normalise_span_ids(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            raise ValueError("source_span_ids must be a sequence")
        return tuple(_normalise_identifier(item, required=True) for item in value)  # type: ignore[misc]

    @field_validator("evidence", mode="before")
    @classmethod
    def normalise_evidence(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError("threat evidence must be a sequence")
        return tuple(str(item) for item in value)

    @field_validator("article_reference", mode="before")
    @classmethod
    def validate_article(cls, value: Any) -> str:
        article = str(value).strip().title().replace("Art.", "Article")
        if article != "Article 15":
            raise ValueError("security threats in this model must reference Article 15")
        return article


class Article12Logging(_EvidenceModel):
    article_reference: str = "Article 12"
    trace_id: str
    trace_sha256: str
    automatically_recorded: bool = Field(strict=True)
    timestamps_present: bool = Field(strict=True)
    immutable_receipts: bool = Field(strict=True)
    span_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    receipt_ids: tuple[str, ...] = ()
    logging_owner: str
    retention_statement: str
    event_taxonomy: Mapping[str, Any] = Field(default_factory=dict)
    evidence: tuple[EvidenceDigest, ...] = ()
    gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    readiness: EvidenceReadiness | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "Article12" in data and "article_reference" not in data:
            data["article_reference"] = data.pop("Article12")
        if "trace_hash" in data and "trace_sha256" not in data:
            data["trace_sha256"] = data.pop("trace_hash")
        return data

    @field_validator("article_reference")
    @classmethod
    def validate_article(cls, value: str) -> str:
        if value != "Article 12":
            raise ValueError("article_reference must be 'Article 12'")
        return value

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_sha256", mode="before")
    @classmethod
    def normalise_trace_hash(cls, value: Any) -> str:
        digest = _normalise_hash(value, required=True)
        assert digest is not None
        return digest

    @field_validator("receipt_ids", mode="before")
    @classmethod
    def normalise_receipt_ids(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            raise ValueError("receipt_ids must be a sequence")
        return tuple(_normalise_identifier(item, required=True) for item in value)  # type: ignore[misc]

    @field_validator("logging_owner", "retention_statement", mode="before")
    @classmethod
    def normalise_required_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Article 12 ownership and retention statements are required")
        return text

    @field_validator("event_taxonomy", mode="before")
    @classmethod
    def normalise_taxonomy(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("event_taxonomy must be a mapping")
        return value

    @field_validator("gaps", "limitations", mode="before")
    @classmethod
    def normalise_text_sequence(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("value must be a sequence of strings")
        return tuple(str(item).strip() for item in value if str(item).strip())

    @model_validator(mode="after")
    def derive_readiness(self) -> Article12Logging:
        object.__setattr__(self, "event_taxonomy", _freeze_value(self.event_taxonomy))
        if self.readiness is None:
            complete = (
                self.automatically_recorded
                and self.timestamps_present
                and self.immutable_receipts
                and self.span_count > 0
            )
            object.__setattr__(
                self,
                "readiness",
                EvidenceReadiness(
                    status="evidenced" if complete else "partial",
                    score=1.0 if complete else 0.5,
                    rationale=(
                        "The supplied trace has timestamped spans and immutable receipts."
                        if complete
                        else "One or more Article 12 logging controls remain unevidenced."
                    ),
                    evidence=self.evidence,
                    gaps=self.gaps,
                ),
            )
        return self

    @property
    def ready(self) -> bool:
        return self.readiness is not None and self.readiness.status == "evidenced"


class Article14Oversight(_EvidenceModel):
    article_reference: str = "Article 14"
    trace_id: str
    trace_sha256: str
    human_oversight_documented: bool = Field(strict=True)
    reviewer_identified: bool = Field(strict=True)
    intervention_criteria_documented: bool = Field(strict=True)
    oversight_mechanism: str | None = None
    reviewer_role: str | None = None
    competence_record: str | None = None
    escalation_procedure: str | None = None
    human_review_signoff: HumanReviewSignoff | None = None
    human_reviews: tuple[HumanReviewSignoff, ...] = ()
    evidence: tuple[EvidenceDigest, ...] = ()
    gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    readiness: EvidenceReadiness | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "article14" in data and "article_reference" not in data:
            data["article_reference"] = data.pop("article14")
        for alias in ("signoff", "review_signoff", "human_review"):
            if alias in data and "human_review_signoff" not in data:
                data["human_review_signoff"] = data.pop(alias)
        return data

    @field_validator("article_reference")
    @classmethod
    def validate_article(cls, value: str) -> str:
        if value != "Article 14":
            raise ValueError("article_reference must be 'Article 14'")
        return value

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_sha256", mode="before")
    @classmethod
    def normalise_trace_hash(cls, value: Any) -> str:
        digest = _normalise_hash(value, required=True)
        assert digest is not None
        return digest

    @field_validator("human_reviews", mode="before")
    @classmethod
    def normalise_reviews(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, Mapping, HumanReviewSignoff)):
            raise ValueError("human_reviews must be a sequence")
        return tuple(value)

    @field_validator("gaps", "limitations", mode="before")
    @classmethod
    def normalise_text_sequence(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("value must be a sequence")
        return tuple(str(item).strip() for item in value if str(item).strip())

    @model_validator(mode="after")
    def derive_readiness(self) -> Article14Oversight:
        if self.human_review_signoff is not None and not self.human_reviews:
            object.__setattr__(self, "human_reviews", (self.human_review_signoff,))
        if self.readiness is None:
            complete = (
                self.human_oversight_documented
                and self.reviewer_identified
                and self.intervention_criteria_documented
                and bool(self.human_reviews)
            )
            object.__setattr__(
                self,
                "readiness",
                EvidenceReadiness(
                    status="evidenced" if complete else "partial",
                    score=1.0 if complete else 0.5,
                    rationale=(
                        "Named human oversight and a cryptographically signed review are supplied."
                        if complete
                        else "Article 14 oversight evidence is incomplete."
                    ),
                    evidence=self.evidence,
                    gaps=self.gaps,
                ),
            )
        return self

    @property
    def ready(self) -> bool:
        return self.readiness is not None and self.readiness.status == "evidenced"


class Article15Security(_EvidenceModel):
    article_reference: str = "Article 15"
    trace_id: str
    trace_sha256: str
    robustness_documented: bool = Field(strict=True)
    cybersecurity_documented: bool = Field(strict=True)
    prompt_injection_detected: bool = Field(strict=True)
    prompt_injection_caught: bool = Field(strict=True)
    threats: tuple[SecurityThreat, ...] = ()
    cybersecurity_controls: tuple[str, ...] = ()
    robustness_tests: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = Field(default_factory=dict)
    evidence: tuple[EvidenceDigest, ...] = ()
    gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    readiness: EvidenceReadiness | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "article15" in data and "article_reference" not in data:
            data["article_reference"] = data.pop("article15")
        if "robustness_tested" in data and "robustness_documented" not in data:
            data["robustness_documented"] = data.pop("robustness_tested")
        if "cybersecurity_tested" in data and "cybersecurity_documented" not in data:
            data["cybersecurity_documented"] = data.pop("cybersecurity_tested")
        if "injection_detected" in data and "prompt_injection_detected" not in data:
            data["prompt_injection_detected"] = data.pop("injection_detected")
        if "injection_caught" in data and "prompt_injection_caught" not in data:
            data["prompt_injection_caught"] = data.pop("injection_caught")
        return data

    @field_validator("article_reference")
    @classmethod
    def validate_article(cls, value: str) -> str:
        if value != "Article 15":
            raise ValueError("article_reference must be 'Article 15'")
        return value

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_sha256", mode="before")
    @classmethod
    def normalise_trace_hash(cls, value: Any) -> str:
        digest = _normalise_hash(value, required=True)
        assert digest is not None
        return digest

    @field_validator("threats", "cybersecurity_controls", "robustness_tests", mode="before")
    @classmethod
    def normalise_sequences(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError("value must be a sequence")
        return tuple(value)

    @field_validator("metrics", mode="before")
    @classmethod
    def normalise_metrics(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("security metrics must be a mapping")
        return value

    @field_validator("gaps", "limitations", mode="before")
    @classmethod
    def normalise_text_sequence(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("value must be a sequence")
        return tuple(str(item).strip() for item in value if str(item).strip())

    @model_validator(mode="after")
    def validate_security_relationships(self) -> Article15Security:
        if self.prompt_injection_caught and not self.prompt_injection_detected:
            raise ValueError("prompt_injection_caught requires prompt_injection_detected")
        object.__setattr__(self, "metrics", _freeze_value(self.metrics))
        if self.readiness is None:
            complete = (
                self.robustness_documented
                and self.cybersecurity_documented
                and not self.gaps
            )
            object.__setattr__(
                self,
                "readiness",
                EvidenceReadiness(
                    status="evidenced" if complete else "partial",
                    score=1.0 if complete else 0.5,
                    rationale=(
                        "The supplied engineering evidence addresses robustness and cybersecurity."
                        if complete
                        else "Article 15 evidence has unresolved engineering gaps."
                    ),
                    evidence=self.evidence,
                    gaps=self.gaps,
                ),
            )
        return self

    @property
    def ready(self) -> bool:
        return self.readiness is not None and self.readiness.status == "evidenced"


class SystemLimitations(_EvidenceModel):
    constitutional_rule: str = CONSTITUTIONAL_RULE
    not_conformity_assessment: bool = True
    counsel_classifies: bool = True
    counsel_review_required: bool = True
    scope: str = "Immutable engineering evidence for Articles 12, 14, and 15 only."
    statements: tuple[str, ...] = (
        "This exhibit does not determine legal classification or conformity.",
        "Cryptographic integrity does not establish truth, admissibility, or legal effect.",
        "Current harmonisation and Official Journal status must be verified by counsel.",
    )
    generated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        for alias in ("limitations", "items", "limitations_statements"):
            if alias in data and "statements" not in data:
                data["statements"] = data.pop(alias)
        return data

    @field_validator("constitutional_rule")
    @classmethod
    def validate_rule(cls, value: str) -> str:
        if value != CONSTITUTIONAL_RULE:
            raise ValueError(f"constitutional_rule must be '{CONSTITUTIONAL_RULE}'")
        return value

    @field_validator("scope", mode="before")
    @classmethod
    def normalise_scope(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("limitations scope must not be empty")
        return text

    @field_validator("statements", mode="before")
    @classmethod
    def normalise_statements(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("statements must be a sequence")
        statements = tuple(str(item).strip() for item in value if str(item).strip())
        if not statements:
            raise ValueError("at least one limitation statement is required")
        return statements

    @field_validator("generated_at", mode="before")
    @classmethod
    def normalise_generated_at(cls, value: Any) -> datetime | None:
        return _normalise_timestamp(value)

    @model_validator(mode="after")
    def validate_constitutional_invariants(self) -> SystemLimitations:
        if not self.not_conformity_assessment or not self.counsel_classifies:
            raise ValueError("the constitutional non-conformity limitation cannot be disabled")
        return self


class ExhibitPack(_EvidenceModel):
    schema_version: str = "1.0.0"
    pack_id: str = Field(validation_alias=AliasChoices("pack_id", "exhibit_id"))
    job_id: str
    trace_id: str
    trace_sha256: str
    trace_format: TraceFormat = TraceFormat.OPENINFERENCE
    generated_at: datetime
    article_12_logging: Article12Logging = Field(
        validation_alias=AliasChoices(
            "article_12_logging",
            "article_12",
            "logging",
            "Article12Logging",
        )
    )
    article_14_oversight: Article14Oversight = Field(
        validation_alias=AliasChoices(
            "article_14_oversight",
            "article_14",
            "oversight",
            "Article14Oversight",
        )
    )
    article_15_security: Article15Security = Field(
        validation_alias=AliasChoices(
            "article_15_security",
            "article_15",
            "security",
            "Article15Security",
        )
    )
    human_review_signoff: HumanReviewSignoff | None = Field(
        default=None,
        validation_alias=AliasChoices("human_review_signoff", "human_review", "review"),
    )
    human_reviews: tuple[HumanReviewSignoff, ...] = ()
    standards_catalog_id: str | None = None
    standards: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[EvidenceDigest, ...] = ()
    system_limitations: SystemLimitations = Field(
        default_factory=SystemLimitations,
        validation_alias=AliasChoices("system_limitations", "limitations"),
    )
    constitutional_rule: str = CONSTITUTIONAL_RULE
    pack_sha256: str | None = Field(
        default=None,
        validation_alias=AliasChoices("pack_sha256", "exhibit_sha256", "hash"),
    )

    @field_validator("pack_id", "job_id", mode="before")
    @classmethod
    def normalise_identifiers(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_id", mode="before")
    @classmethod
    def normalise_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("trace_sha256", "pack_sha256", mode="before")
    @classmethod
    def normalise_hash(cls, value: Any) -> str | None:
        return _normalise_hash(value)

    @field_validator("generated_at", mode="before")
    @classmethod
    def normalise_generated_at(cls, value: Any) -> datetime:
        timestamp = _normalise_timestamp(value, required=True)
        assert timestamp is not None
        return timestamp

    @field_validator("standards_catalog_id", mode="before")
    @classmethod
    def normalise_catalog_id(cls, value: Any) -> str | None:
        return _normalise_identifier(value)

    @field_validator("human_reviews", "standards", mode="before")
    @classmethod
    def normalise_sequences(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError("value must be a sequence")
        return tuple(value)

    @field_validator("constitutional_rule")
    @classmethod
    def validate_constitutional_rule(cls, value: str) -> str:
        if value != CONSTITUTIONAL_RULE:
            raise ValueError(f"constitutional_rule must be '{CONSTITUTIONAL_RULE}'")
        return value

    @model_validator(mode="after")
    def validate_pack(self) -> ExhibitPack:
        articles = (
            self.article_12_logging,
            self.article_14_oversight,
            self.article_15_security,
        )
        for article in articles:
            if article.trace_id != self.trace_id or article.trace_sha256 != self.trace_sha256:
                raise ValueError("article evidence must match the pack trace identity and digest")
        if self.human_review_signoff is not None and not self.human_reviews:
            object.__setattr__(self, "human_reviews", (self.human_review_signoff,))
        object.__setattr__(
            self,
            "standards",
            tuple(_freeze_value(item) for item in self.standards),
        )

        unsigned = self.model_dump(mode="python")
        unsigned.pop("pack_sha256", None)
        expected_hash = sha256_hex(unsigned)
        if self.pack_sha256 is not None and self.pack_sha256 != expected_hash:
            raise ValueError("pack_sha256 does not match canonical pack evidence")
        object.__setattr__(self, "pack_sha256", expected_hash)
        return self

    @property
    def article_12(self) -> Article12Logging:
        return self.article_12_logging

    @property
    def article_14(self) -> Article14Oversight:
        return self.article_14_oversight

    @property
    def article_15(self) -> Article15Security:
        return self.article_15_security

    @property
    def evidence_sha256(self) -> str:
        return self.pack_sha256 or sha256_hex(self.model_dump(mode="python"))