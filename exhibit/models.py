from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
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
_TRACE_ID_RE = re.compile(r"^[0-9a-f]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _FrozenDict(dict[str, Any]):
    """A shallow dict that prevents accidental mutation of evidence."""

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
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=lambda item: str(item)))
    if isinstance(value, frozenset):
        return tuple(sorted((_freeze_value(item) for item in value), key=lambda item: str(item)))
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
    ).encode("utf-8")


def sha256_hex(value: bytes | bytearray | memoryview | str | Any) -> str:
    """Hash bytes/string directly and all other values through canonical JSON."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        digest_input = bytes(value)
    elif isinstance(value, str):
        digest_input = value.encode("utf-8")
    else:
        digest_input = canonical_json_bytes(value)
    return hashlib.sha256(digest_input).hexdigest()


def _normalise_identifier(value: Any, *, width: int, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError("identifier is required")
        return None
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid identifiers")
    if isinstance(value, int):
        text = format(value, "x")
    elif isinstance(value, str):
        text = value.strip().lower()
    else:
        raise ValueError("identifier must be a string or integer")

    if text.startswith("0x"):
        text = text[2:]

    if not text or (text and set(text) == {"0"}):
        if required:
            raise ValueError("identifier cannot be empty or all zero")
        return None

    if _TRACE_ID_RE.fullmatch(text) and len(text) < width:
        text = text.zfill(width)
    return text


def _coerce_epoch_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if not isinstance(value, (str, int, float, Decimal)):
        return value

    text = str(value).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return value

    try:
        numeric = Decimal(text)
    except Exception:
        return value
    if not numeric.is_finite():
        raise ValueError("timestamps must be finite")

    absolute = abs(numeric)
    if absolute >= Decimal("1e17"):
        return numeric / Decimal("1e9")
    if absolute >= Decimal("1e14"):
        return numeric / Decimal("1e6")
    if absolute >= Decimal("1e11"):
        return numeric / Decimal("1e3")
    return numeric


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalise_status_code(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("boolean status codes are not supported")
    if isinstance(value, int):
        if value in (0, 1, 2):
            return value
        raise ValueError("OpenTelemetry status code must be 0, 1, or 2")
    if isinstance(value, str):
        text = value.strip().upper().replace("-", "_").replace(" ", "_")
        if text.startswith("STATUS_CODE_"):
            text = text.removeprefix("STATUS_CODE_")
        aliases = {
            "UNSET": 0,
            "UNKNOWN": 0,
            "OK": 1,
            "SUCCESS": 1,
            "ERROR": 2,
            "ERR": 2,
        }
        if text in aliases:
            return aliases[text]
    return value


class SpanKind(str, Enum):
    UNSPECIFIED = "unspecified"
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TraceFormat(str, Enum):
    OPENINFERENCE_JSONL = "openinference-jsonl"
    OPENINFERENCE_JSON = "openinference-json"
    OTLP_JSON = "otlp-json"
    OTEL_JSON = "otel-json"
    JSONL = "jsonl"
    JSON = "json"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
    )


class SpanEvent(_FrozenModel):
    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "event_name", "eventName"),
    )
    timestamp: datetime = Field(
        validation_alias=AliasChoices(
            "timestamp",
            "time",
            "timestampUnixNano",
            "timestamp_unix_nano",
            "timeUnixNano",
            "time_unix_nano",
        ),
    )
    attributes: Mapping[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("attributes", "attrs"),
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_event(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        payload.pop("droppedAttributesCount", None)
        payload.pop("dropped_attributes_count", None)
        for key in ("timestamp", "timestampUnixNano", "timestamp_unix_nano", "timeUnixNano", "time_unix_nano"):
            if key in payload:
                payload[key] = _coerce_epoch_value(payload[key])
        return payload

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("event name must not be empty")
        return text

    @field_validator("timestamp", mode="after")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        normalised = _ensure_utc(value)
        assert normalised is not None
        return normalised

    @field_validator("attributes", mode="after")
    @classmethod
    def freeze_attributes(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_value(dict(value))

    @property
    def canonical_sha256(self) -> str:
        return sha256_hex(self.model_dump(mode="python"))

    @property
    def sha256(self) -> str:
        return self.canonical_sha256


class SpanLink(_FrozenModel):
    trace_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("trace_id", "traceId", "linkedTraceId", "linked_trace_id"),
    )
    span_id: str = Field(
        validation_alias=AliasChoices("span_id", "spanId", "linkedSpanId", "linked_span_id", "id"),
    )
    attributes: Mapping[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("attributes", "attrs"),
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_link(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        payload.pop("droppedAttributesCount", None)
        payload.pop("dropped_attributes_count", None)
        return payload

    @field_validator("trace_id", mode="before")
    @classmethod
    def validate_trace_id(cls, value: Any) -> str | None:
        return _normalise_identifier(value, width=32, required=False)

    @field_validator("span_id", mode="before")
    @classmethod
    def validate_span_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=16, required=True)
        assert identifier is not None
        return identifier

    @field_validator("attributes", mode="after")
    @classmethod
    def freeze_attributes(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_value(dict(value))

    @property
    def canonical_sha256(self) -> str:
        return sha256_hex(self.model_dump(mode="python"))

    @property
    def sha256(self) -> str:
        return self.canonical_sha256


class Span(_FrozenModel):
    trace_id: str = Field(
        validation_alias=AliasChoices("trace_id", "traceId", "traceID"),
    )
    span_id: str = Field(
        validation_alias=AliasChoices("span_id", "spanId", "spanID", "id"),
    )
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
        validation_alias=AliasChoices("name", "span_name", "spanName", "operation"),
    )
    kind: SpanKind = Field(
        default=SpanKind.INTERNAL,
        validation_alias=AliasChoices("kind", "span_kind", "spanKind"),
    )
    start_time: datetime = Field(
        validation_alias=AliasChoices(
            "start_time",
            "startTime",
            "start",
            "timestamp",
            "start_time_unix_nano",
            "startTimeUnixNano",
        ),
    )
    end_time: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "end_time",
            "endTime",
            "end",
            "end_time_unix_nano",
            "endTimeUnixNano",
        ),
    )
    status: SpanStatus = Field(default=SpanStatus.UNSET)
    status_message: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "status_message",
            "statusMessage",
            "status_description",
            "statusDescription",
        ),
    )
    attributes: Mapping[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("attributes", "attrs"),
    )
    events: tuple[SpanEvent, ...] = Field(default_factory=tuple)
    links: tuple[SpanLink, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def prepare_span(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="python", by_alias=True)
        if not isinstance(value, Mapping):
            return value

        payload = dict(value)

        status = payload.get("status")
        if isinstance(status, Mapping):
            code = status.get("code", status.get("status_code", status.get("statusCode")))
            if code is not None:
                payload["status"] = code
            message = status.get(
                "message",
                status.get(
                    "description",
                    status.get("status_message", status.get("statusMessage")),
                ),
            )
            if message is not None and not any(
                key in payload
                for key in ("status_message", "statusMessage", "status_description", "statusDescription")
            ):
                payload["status_message"] = message

        for key in (
            "start_time",
            "startTime",
            "start",
            "timestamp",
            "start_time_unix_nano",
            "startTimeUnixNano",
            "end_time",
            "endTime",
            "end",
            "end_time_unix_nano",
            "endTimeUnixNano",
        ):
            if key in payload:
                payload[key] = _coerce_epoch_value(payload[key])

        return payload

    @field_validator("trace_id", mode="before")
    @classmethod
    def validate_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("span_id", mode="before")
    @classmethod
    def validate_span_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=16, required=True)
        assert identifier is not None
        return identifier

    @field_validator("parent_span_id", mode="before")
    @classmethod
    def validate_parent_span_id(cls, value: Any) -> str | None:
        return _normalise_identifier(value, width=16, required=False)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("span name must not be empty")
        return text

    @field_validator("kind", mode="before")
    @classmethod
    def validate_kind(cls, value: Any) -> Any:
        if isinstance(value, SpanKind):
            return value
        if isinstance(value, bool):
            raise ValueError("boolean span kinds are not supported")
        if isinstance(value, int):
            mapping = {
                0: SpanKind.UNSPECIFIED,
                1: SpanKind.INTERNAL,
                2: SpanKind.SERVER,
                3: SpanKind.CLIENT,
                4: SpanKind.PRODUCER,
                5: SpanKind.CONSUMER,
            }
            if value not in mapping:
                raise ValueError("OpenTelemetry span kind must be between 0 and 5")
            return mapping[value]
        if isinstance(value, str):
            text = value.strip().upper().replace("-", "_").replace(" ", "_")
            if text.startswith("SPAN_KIND_"):
                text = text.removeprefix("SPAN_KIND_")
            aliases = {
                "UNSPECIFIED": SpanKind.UNSPECIFIED,
                "INTERNAL": SpanKind.INTERNAL,
                "SERVER": SpanKind.SERVER,
                "CLIENT": SpanKind.CLIENT,
                "PRODUCER": SpanKind.PRODUCER,
                "CONSUMER": SpanKind.CONSUMER,
            }
            if text in aliases:
                return aliases[text]
        return value

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, value: Any) -> Any:
        if isinstance(value, SpanStatus):
            return value
        return _normalise_status_code(value)

    @field_validator("status_message", mode="before")
    @classmethod
    def validate_status_message(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def validate_times(cls, value: datetime | None) -> datetime | None:
        return _ensure_utc(value)

    @field_validator("attributes", mode="after")
    @classmethod
    def freeze_attributes(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_value(dict(value))

    @model_validator(mode="after")
    def validate_interval_and_parent(self) -> Span:
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        if self.parent_span_id == self.span_id:
            raise ValueError("a span cannot be its own parent")
        return self

    def attribute(self, key: str, default: Any = None) -> Any:
        if key in self.attributes:
            return self.attributes[key]
        folded = key.casefold()
        for candidate, value in self.attributes.items():
            if candidate.casefold() == folded:
                return value
        return default

    @staticmethod
    def _semantic_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            return value
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return value

    @property
    def id(self) -> str:
        return self.span_id

    @property
    def parent_id(self) -> str | None:
        return self.parent_span_id

    @property
    def input_value(self) -> Any:
        return self._semantic_value(self.attribute("input.value"))

    @property
    def output_value(self) -> Any:
        return self._semantic_value(self.attribute("output.value"))

    @property
    def openinference_span_kind(self) -> str | None:
        value = self.attribute("openinference.span.kind")
        return str(value).strip().upper() if value is not None else None

    @property
    def is_openinference(self) -> bool:
        if self.attribute("openinference.span.kind") is not None:
            return True
        return any(
            key.startswith(("openinference.", "llm.", "retrieval.", "embedding.", "tool."))
            for key in self.attributes
        )

    @property
    def status_code(self) -> int:
        return {
            SpanStatus.UNSET: 0,
            SpanStatus.OK: 1,
            SpanStatus.ERROR: 2,
        }[self.status]

    @property
    def status_description(self) -> str | None:
        return self.status_message

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000.0

    @property
    def duration(self) -> timedelta:
        if self.end_time is None:
            return timedelta(0)
        return self.end_time - self.start_time

    @property
    def canonical_sha256(self) -> str:
        return sha256_hex(self.model_dump(mode="python"))

    @property
    def sha256(self) -> str:
        return self.canonical_sha256

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class Trace(_FrozenModel):
    trace_id: str
    spans: tuple[Span, ...] = Field(min_length=1)
    source_format: TraceFormat = Field(
        default=TraceFormat.OPENINFERENCE_JSONL,
        validation_alias=AliasChoices("source_format", "sourceFormat"),
    )
    source_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_name", "sourceName", "source_path", "sourcePath"),
    )
    source_sha256: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_sha256", "sourceSha256", "source_hash"),
    )
    issues: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("issues", "structural_issues", "warnings"),
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_trace(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="python", by_alias=True)
        if not isinstance(value, Mapping):
            return value

        payload = dict(value)
        spans_value = payload.get("spans", ())
        if isinstance(spans_value, Sequence) and not isinstance(spans_value, (str, bytes, bytearray)):
            spans = list(spans_value)
        else:
            spans = []
        payload["spans"] = spans

        for alias in ("structural_issues", "warnings", "integrity_issues"):
            if alias in payload:
                issues = payload.get("issues", ())
                if isinstance(issues, str):
                    issues = (issues,)
                payload["issues"] = tuple(issues) + tuple(payload.pop(alias))
                break

        source_format = payload.get("source_format", payload.get("sourceFormat"))
        if isinstance(source_format, str):
            folded = source_format.strip().lower().replace("_", "-")
            aliases = {
                "openinference": TraceFormat.OPENINFERENCE_JSONL,
                "openinference-jsonl": TraceFormat.OPENINFERENCE_JSONL,
                "openinference-json": TraceFormat.OPENINFERENCE_JSON,
                "otlp": TraceFormat.OTLP_JSON,
                "otlp-json": TraceFormat.OTLP_JSON,
                "otel": TraceFormat.OTEL_JSON,
                "otel-json": TraceFormat.OTEL_JSON,
                "jsonl": TraceFormat.JSONL,
                "json": TraceFormat.JSON,
            }
            if folded in aliases:
                payload["source_format"] = aliases[folded]

        supplied_trace_ids = {
            span.trace_id if isinstance(span, Span) else str(
                span.get("trace_id", span.get("traceId"))
            ).lower()
            for span in spans
            if (
                isinstance(span, Span)
                or (isinstance(span, Mapping) and (span.get("trace_id") is not None or span.get("traceId") is not None))
            )
        }
        if payload.get("trace_id") is None and len(supplied_trace_ids) == 1:
            payload["trace_id"] = next(iter(supplied_trace_ids))
        return payload

    @field_validator("trace_id", mode="before")
    @classmethod
    def validate_trace_id(cls, value: Any) -> str:
        identifier = _normalise_identifier(value, width=32, required=True)
        assert identifier is not None
        return identifier

    @field_validator("source_name", mode="before")
    @classmethod
    def validate_source_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("source_sha256", mode="before")
    @classmethod
    def validate_source_sha256(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not _SHA256_RE.fullmatch(text):
            raise ValueError("source_sha256 must be a 64-character hexadecimal SHA-256 digest")
        return text

    @model_validator(mode="after")
    def validate_graph(self) -> Trace:
        by_id: dict[str, Span] = {}
        input_order: dict[str, int] = {}
        for index, span in enumerate(self.spans):
            if span.trace_id != self.trace_id:
                raise ValueError(
                    f"span '{span.span_id}' belongs to trace '{span.trace_id}', not '{self.trace_id}'"
                )
            if span.span_id in by_id:
                raise ValueError(f"duplicate span identifier '{span.span_id}'")
            by_id[span.span_id] = span
            input_order[span.span_id] = index

        missing_parent: dict[str, str | None] = {}
        issues = list(self.issues)
        for span in self.spans:
            if span.parent_span_id is not None and span.parent_span_id not in by_id:
                message = (
                    f"parent span '{span.parent_span_id}' not found for span "
                    f"'{span.span_id}'; treated as a root"
                )
                missing_parent[span.span_id] = None
                if message not in issues:
                    issues.append(message)
            else:
                missing_parent[span.span_id] = span.parent_span_id

        state: dict[str, int] = {}

        def visit(span_id: str, path: tuple[str, ...]) -> None:
            current_state = state.get(span_id, 0)
            if current_state == 1:
                cycle_start = path.index(span_id) if span_id in path else 0
                cycle = path[cycle_start:] + (span_id,)
                raise ValueError(f"parent relationship contains a cycle: {' -> '.join(cycle)}")
            if current_state == 2:
                return
            state[span_id] = 1
            parent = missing_parent.get(span_id)
            if parent is not None and parent in by_id:
                visit(parent, path + (span_id,))
            state[span_id] = 2

        for span_id in by_id:
            visit(span_id, ())

        effective_parent = missing_parent
        children: dict[str | None, list[str]] = {}
        for span in self.spans:
            parent = effective_parent[span.span_id]
            children.setdefault(parent, []).append(span.span_id)

        for child_ids in children.values():
            child_ids.sort(key=lambda item: input_order[item])

        ordered_ids: list[str] = []

        def append_branch(span_id: str) -> None:
            if span_id in ordered_ids:
                return
            ordered_ids.append(span_id)
            for child_id in children.get(span_id, ()):
                append_branch(child_id)

        for root_id in children.get(None, ()):
            append_branch(root_id)
        for span in self.spans:
            append_branch(span.span_id)

        object.__setattr__(self, "issues", tuple(issues))
        object.__setattr__(self, "spans", tuple(by_id[span_id] for span_id in ordered_ids))
        return self

    @property
    def root_spans(self) -> tuple[Span, ...]:
        present = {span.span_id for span in self.spans}
        return tuple(
            span
            for span in self.spans
            if span.parent_span_id is None or span.parent_span_id not in present
        )

    @property
    def roots(self) -> tuple[Span, ...]:
        return self.root_spans

    @property
    def root_span_ids(self) -> tuple[str, ...]:
        return tuple(span.span_id for span in self.root_spans)

    @property
    def span_by_id(self) -> Mapping[str, Span]:
        return _FrozenDict({span.span_id: span for span in self.spans})

    @property
    def span_hashes(self) -> Mapping[str, str]:
        return _FrozenDict({span.span_id: span.canonical_sha256 for span in self.spans})

    @property
    def flattened_spans(self) -> tuple[Span, ...]:
        return self.spans

    @property
    def orphaned_span_ids(self) -> tuple[str, ...]:
        present = {span.span_id for span in self.spans}
        return tuple(
            span.span_id
            for span in self.spans
            if span.parent_span_id is not None and span.parent_span_id not in present
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.issues

    @property
    def structural_issues(self) -> tuple[str, ...]:
        return self.issues

    @property
    def integrity_issues(self) -> tuple[str, ...]:
        return self.issues

    @property
    def structurally_valid(self) -> bool:
        return not self.orphaned_span_ids

    @property
    def start_time(self) -> datetime:
        return min(span.start_time for span in self.spans)

    @property
    def end_time(self) -> datetime:
        return max(span.end_time or span.start_time for span in self.spans)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time).total_seconds() * 1000.0

    @property
    def canonical_sha256(self) -> str:
        return sha256_hex(self.model_dump(mode="python"))

    @property
    def sha256(self) -> str:
        return self.canonical_sha256

    def walk(self) -> Any:
        children: dict[str | None, list[Span]] = {}
        present = {span.span_id for span in self.spans}
        for span in self.spans:
            parent = (
                span.parent_span_id
                if span.parent_span_id is not None and span.parent_span_id in present
                else None
            )
            children.setdefault(parent, []).append(span)

        def descend(span: Span) -> Any:
            yield span
            for child in children.get(span.span_id, ()):
                yield from descend(child)

        for root in children.get(None, ()):
            yield from descend(root)

    def get_span(self, span_id: str) -> Span:
        requested = str(span_id).strip().lower()
        if requested.startswith("0x"):
            requested = requested[2:]
        for span in self.spans:
            if span.span_id == requested:
                return span
        raise KeyError(span_id)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


__all__ = [
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanLink",
    "SpanStatus",
    "Trace",
    "TraceFormat",
    "canonical_json_bytes",
    "sha256_hex",
]