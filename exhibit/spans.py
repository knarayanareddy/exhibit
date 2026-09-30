from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from pydantic import ValidationError

from .models import (
    Span,
    SpanEvent,
    SpanKind,
    SpanLink,
    SpanStatus,
    Trace,
    TraceFormat,
    canonical_json_bytes,
    sha256_hex,
)


UTC = timezone.utc

_TRACE_ID_KEYS = ("trace_id", "traceId", "traceID")
_SPAN_ID_KEYS = ("span_id", "spanId", "spanID", "id")
_PARENT_ID_KEYS = (
    "parent_span_id",
    "parentSpanId",
    "parent_span_ID",
    "parent_id",
    "parentId",
)
_NAME_KEYS = ("name", "span_name", "spanName", "operation")
_START_KEYS = (
    ("startTimeUnixNano", "nanosecond"),
    ("start_time_unix_nano", "nanosecond"),
    ("startTimeUnixMicro", "microsecond"),
    ("startTimeUnixMicroNano", "nanosecond"),
    ("startTimeUnixMilli", "millisecond"),
    ("startTime", None),
    ("start_time", None),
    ("start", None),
    ("timestamp", None),
)
_END_KEYS = (
    ("endTimeUnixNano", "nanosecond"),
    ("end_time_unix_nano", "nanosecond"),
    ("endTimeUnixMicro", "microsecond"),
    ("endTimeUnixMilli", "millisecond"),
    ("endTime", None),
    ("end_time", None),
    ("end", None),
)


@dataclass(frozen=True, slots=True)
class ParseIssue:
    source: str
    record: int
    code: str
    reason: str

    def __str__(self) -> str:
        return f"{self.source}:{self.record}: {self.code}: {self.reason}"


class SpanParseError(ValueError):
    """A source record could not be converted into trustworthy span evidence."""

    def __init__(self, issue: ParseIssue):
        self.issue = issue
        super().__init__(str(issue))


class DuplicateSpanError(SpanParseError):
    """Two non-equivalent records claim the same trace/span identifier."""


class EmptyTraceError(ValueError):
    """No executable span evidence was found in the supplied source."""


class TraceNotFoundError(ValueError):
    """The requested trace identifier is absent from an ingestion result."""

    def __init__(self, trace_id: str, available: Sequence[str] = ()):
        self.trace_id = trace_id
        self.available = tuple(available)
        available_text = ", ".join(self.available) if self.available else "none"
        super().__init__(f"trace '{trace_id}' not found; available traces: {available_text}")


class AmbiguousTraceError(ValueError):
    """A single trace was requested but the source contains several traces."""

    def __init__(self, available: Sequence[str]):
        self.available = tuple(available)
        super().__init__(
            "source contains multiple traces; select one explicitly: " + ", ".join(self.available)
        )


@dataclass(frozen=True, slots=True)
class IngestionResult:
    source_name: str
    source_sha256: str
    traces: tuple[Trace, ...]
    issues: tuple[ParseIssue, ...] = ()

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.traces)

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, index: int) -> Trace:
        return self.traces[index]

    @property
    def trace_ids(self) -> tuple[str, ...]:
        return tuple(trace.trace_id for trace in self.traces)

    @property
    def spans(self) -> tuple[Span, ...]:
        return tuple(span for trace in self.traces for span in trace.spans)

    def get_trace(self, trace_id: str) -> Trace:
        requested = str(trace_id).strip().lower()
        if requested.startswith("0x"):
            requested = requested[2:]
        if len(requested) < 32 and all(character in "0123456789abcdef" for character in requested):
            requested = requested.zfill(32)
        for trace in self.traces:
            if trace.trace_id == requested:
                return trace
        raise TraceNotFoundError(trace_id, self.trace_ids)


@dataclass(frozen=True, slots=True)
class _LoadedSource:
    name: str
    sha256: str
    text: str | None = None
    records: tuple[Mapping[str, Any], ...] | None = None
    kind: str = "text"


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key '{key}'")
        result[key] = value
    return result


def _json_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)


def _source_name(value: Any, override: str | None) -> str:
    if override is not None:
        return override
    name = getattr(value, "name", None)
    if name is not None:
        return Path(str(name)).name
    if isinstance(value, (str, os.PathLike)):
        candidate = Path(value)
        if candidate.name:
            return candidate.name
    return "<memory>"


def _hash_records(records: Sequence[Mapping[str, Any]]) -> str:
    try:
        payload: Any = list(records)
        return sha256_hex(payload)
    except (TypeError, ValueError):
        return sha256_hex(repr(records).encode("utf-8"))


def _looks_like_json_text(value: str) -> bool:
    stripped = value.lstrip("\ufeff \t\r\n")
    return not stripped or stripped[0] in "[{\"-0123456789"


def _load_source(source: Any, *, source_name: str | None = None) -> _LoadedSource:
    if isinstance(source, os.PathLike):
        path = Path(source)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise
        return _LoadedSource(
            name=source_name or path.name,
            sha256=sha256_hex(raw),
            text=raw.decode("utf-8-sig"),
        )

    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        return _LoadedSource(
            name=source_name or "<bytes>",
            sha256=sha256_hex(raw),
            text=raw.decode("utf-8-sig"),
        )

    if isinstance(source, str):
        if not _looks_like_json_text(source):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(path)
            raw = path.read_bytes()
            return _LoadedSource(
                name=source_name or path.name,
                sha256=sha256_hex(raw),
                text=raw.decode("utf-8-sig"),
            )
        return _LoadedSource(
            name=source_name or "<string>",
            sha256=sha256_hex(source.encode("utf-8")),
            text=source,
        )

    if isinstance(source, (TextIO, BinaryIO, io.TextIOBase, io.BufferedIOBase)) or hasattr(
        source, "read"
    ):
        original_position: int | None = None
        if hasattr(source, "tell"):
            try:
                original_position = source.tell()
            except (OSError, ValueError):
                original_position = None
        raw_value = source.read()
        if original_position is not None and hasattr(source, "seek"):
            try:
                source.seek(original_position)
            except (OSError, ValueError):
                pass
        raw = raw_value.encode("utf-8") if isinstance(raw_value, str) else bytes(raw_value)
        text = raw.decode("utf-8-sig") if isinstance(raw_value, str) else raw.decode("utf-8-sig")
        return _LoadedSource(
            name=source_name or _source_name(source, None),
            sha256=sha256_hex(raw),
            text=text,
        )

    if isinstance(source, Mapping):
        records = (dict(source),)
        return _LoadedSource(
            name=source_name or "<mapping>",
            sha256=_hash_records(records),
            records=records,
            kind="records",
        )

    if isinstance(source, Iterable) and not isinstance(source, (str, bytes, bytearray)):
        records = tuple(dict(record) if isinstance(record, Mapping) else record for record in source)  # type: ignore[arg-type]
        if not all(isinstance(record, Mapping) for record in records):
            raise TypeError("record iterables must contain mappings")
        typed_records = tuple(records)  # type: ignore[assignment]
        return _LoadedSource(
            name=source_name or "<records>",
            sha256=_hash_records(typed_records),
            records=typed_records,
            kind="records",
        )

    raise TypeError(f"unsupported span source type: {type(source).__name__}")


def decode_otlp_value(value: Any) -> Any:
    """Decode an OTLP JSON ``AnyValue`` without requiring the OTLP SDK."""

    if not isinstance(value, Mapping):
        return value

    string_keys = ("stringValue", "string_value")
    bool_keys = ("boolValue", "bool_value")
    int_keys = ("intValue", "int_value")
    double_keys = ("doubleValue", "double_value")
    bytes_keys = ("bytesValue", "bytes_value")
    array_keys = ("arrayValue", "array_value")
    kvlist_keys = ("kvlistValue", "kvlist_value", "kvListValue")

    for key in string_keys:
        if key in value:
            return value[key]

    for key in bool_keys:
        if key in value:
            if not isinstance(value[key], bool):
                raise ValueError(f"OTLP {key} must be boolean")
            return value[key]

    for key in int_keys:
        if key in value:
            try:
                return int(value[key])
            except (TypeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f"OTLP {key} must be an integer") from exc

    for key in double_keys:
        if key in value:
            try:
                number = float(value[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"OTLP {key} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"OTLP {key} must be finite")
            return number

    for key in bytes_keys:
        if key in value:
            encoded = value[key]
            if not isinstance(encoded, str):
                raise ValueError(f"OTLP {key} must be base64 text")
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"OTLP {key} is not valid base64") from exc

    for key in array_keys:
        if key in value:
            array = value[key]
            if array is None:
                return []
            if not isinstance(array, Mapping):
                raise ValueError(f"OTLP {key} must be an object")
            values = array.get("values", [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise ValueError(f"OTLP {key}.values must be an array")
            return [decode_otlp_value(item) for item in values]

    for key in kvlist_keys:
        if key in value:
            kvlist = value[key]
            if kvlist is None:
                return {}
            if not isinstance(kvlist, Mapping):
                raise ValueError(f"OTLP {key} must be an object")
            values = kvlist.get("values", [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise ValueError(f"OTLP {key}.values must be an array")
            decoded_mapping: dict[str, Any] = {}
            for item in values:
                if not isinstance(item, Mapping) or "key" not in item:
                    raise ValueError(f"OTLP {key} entries require a key")
                decoded_mapping[str(item["key"])] = decode_otlp_value(item.get("value"))
            return decoded_mapping

    return dict(value)


def decode_otlp_attributes(attributes: Any) -> dict[str, Any]:
    """Decode OTLP's list-of-key/value attributes or preserve a plain mapping."""

    if attributes is None:
        return {}
    if isinstance(attributes, Mapping):
        return {str(key): decode_otlp_value(value) for key, value in attributes.items()}
    if not isinstance(attributes, Sequence) or isinstance(attributes, (str, bytes, bytearray)):
        raise ValueError("OTLP attributes must be an array or object")

    decoded: dict[str, Any] = {}
    for index, attribute in enumerate(attributes):
        if not isinstance(attribute, Mapping):
            raise ValueError(f"OTLP attribute at index {index} must be an object")
        if "key" not in attribute:
            raise ValueError(f"OTLP attribute at index {index} has no key")
        key = str(attribute["key"])
        if not key:
            raise ValueError(f"OTLP attribute at index {index} has an empty key")
        decoded[key] = decode_otlp_value(attribute.get("value"))
    return decoded


def _merge_attributes(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        if layer:
            merged.update(layer)
    return merged


def _decode_otlp_identifier(value: Any, *, width: int) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, int):
        return format(value, "x").zfill(width)
    if not isinstance(value, str):
        return value

    text = value.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if text and all(character in "0123456789abcdef" for character in text):
        return text.zfill(width) if len(text) < width else text

    expected_encoded_lengths = {16, 24} if width == 32 else {12, 16}
    if len(text) in expected_encoded_lengths:
        try:
            decoded = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            return text
        if len(decoded) * 2 == width:
            return decoded.hex()
    return text


def _extract_otlp_spans(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes, bytearray, Mapping)):
        result: list[dict[str, Any]] = []
        for item in document:
            if isinstance(item, Mapping):
                result.extend(_extract_otlp_spans(item))
        return result

    resources = document.get("resourceSpans", document.get("resource_spans"))
    if resources is None:
        if any(key in document for key in ("scopeSpans", "scope_spans", "spans")):
            resources = [document]
        else:
            return []

    if not isinstance(resources, Sequence) or isinstance(resources, (str, bytes, bytearray, Mapping)):
        raise ValueError("OTLP resourceSpans must be an array")

    extracted: list[dict[str, Any]] = []
    for resource_record in resources:
        if not isinstance(resource_record, Mapping):
            raise ValueError("each OTLP resourceSpans entry must be an object")
        resource = resource_record.get("resource", {})
        if not isinstance(resource, Mapping):
            raise ValueError("OTLP resource must be an object")
        resource_attributes = decode_otlp_attributes(resource.get("attributes"))

        scopes = resource_record.get("scopeSpans", resource_record.get("scope_spans"))
        if scopes is None:
            scopes = resource_record.get("instrumentationLibrarySpans", [])
        if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes, bytearray, Mapping)):
            raise ValueError("OTLP scopeSpans must be an array")

        for scope_record in scopes:
            if not isinstance(scope_record, Mapping):
                raise ValueError("each OTLP scopeSpans entry must be an object")
            scope = scope_record.get("scope", scope_record.get("instrumentationLibrary", {}))
            scope_attributes = (
                decode_otlp_attributes(scope.get("attributes")) if isinstance(scope, Mapping) else {}
            )
            spans = scope_record.get("spans", [])
            if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray, Mapping)):
                raise ValueError("OTLP scopeSpans.spans must be an array")

            for span_record in spans:
                if not isinstance(span_record, Mapping):
                    raise ValueError("each OTLP span must be an object")
                payload = dict(span_record)
                own_attributes = decode_otlp_attributes(payload.get("attributes"))
                payload["attributes"] = _merge_attributes(
                    resource_attributes,
                    scope_attributes,
                    own_attributes,
                )
                extracted.append(payload)
    return extracted


def _is_otlp_document(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return any(_is_otlp_document(item) for item in value)
    return isinstance(value, Mapping) and any(
        key in value for key in ("resourceSpans", "resource_spans", "scopeSpans", "scope_spans")
    )


def _extract_span_mappings(
    document: Any,
    *,
    inherited_trace_id: Any = None,
    inherited_attributes: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes, bytearray, Mapping)):
        result: list[dict[str, Any]] = []
        for item in document:
            result.extend(
                _extract_span_mappings(
                    item,
                    inherited_trace_id=inherited_trace_id,
                    inherited_attributes=inherited_attributes,
                )
            )
        return result

    if not isinstance(document, Mapping):
        return []

    if _is_otlp_document(document):
        return _extract_otlp_spans(document)

    if "span" in document and isinstance(document["span"], Mapping):
        wrapper = dict(document)
        span = dict(document["span"])
        trace_id = _first_present(document, _TRACE_ID_KEYS, inherited_trace_id)
        if trace_id is not None and not any(key in span for key in _TRACE_ID_KEYS):
            span["trace_id"] = trace_id
        attributes = _merge_attributes(
            inherited_attributes,
            decode_otlp_attributes(document.get("attributes")) if document.get("attributes") is not None else None,
        )
        if attributes:
            span["attributes"] = _merge_attributes(
                attributes,
                decode_otlp_attributes(span.get("attributes")) if span.get("attributes") is not None else None,
            )
        return [span]

    document_trace = _first_present(document, _TRACE_ID_KEYS, inherited_trace_id)
    if "spans" in document:
        spans = document["spans"]
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray, Mapping)):
            raise ValueError("document spans must be an array")
        document_attributes = (
            decode_otlp_attributes(document.get("attributes"))
            if document.get("attributes") is not None
            else None
        )
        result = []
        for item in spans:
            if not isinstance(item, Mapping):
                raise ValueError("each document span must be an object")
            payload = dict(item)
            if document_trace is not None and not any(key in payload for key in _TRACE_ID_KEYS):
                payload["trace_id"] = document_trace
            combined = _merge_attributes(
                inherited_attributes,
                document_attributes,
                decode_otlp_attributes(payload.get("attributes"))
                if payload.get("attributes") is not None
                else None,
            )
            if combined:
                payload["attributes"] = combined
            result.append(payload)
        return result

    for container_key in ("data", "records", "items"):
        if container_key in document and not any(key in document for key in _SPAN_ID_KEYS):
            nested = document[container_key]
            if isinstance(nested, str):
                try:
                    nested = _json_loads(nested)
                except (json.JSONDecodeError, _DuplicateJSONKey):
                    return []
            return _extract_span_mappings(
                nested,
                inherited_trace_id=document_trace,
                inherited_attributes=inherited_attributes,
            )

    if any(key in document for key in _SPAN_ID_KEYS):
        payload = dict(document)
        if document_trace is not None and not any(key in payload for key in _TRACE_ID_KEYS):
            payload["trace_id"] = document_trace
        if inherited_attributes:
            own = (
                decode_otlp_attributes(payload.get("attributes"))
                if payload.get("attributes") is not None
                else None
            )
            payload["attributes"] = _merge_attributes(inherited_attributes, own)
        return [payload]

    return []


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _coerce_datetime(value: Any, unit: str | None = None) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime(value.year, value.month, value.day)
    else:
        if isinstance(value, bool):
            raise ValueError("boolean timestamps are not supported")
        if isinstance(value, (int, float, Decimal)):
            numeric = Decimal(str(value))
            if not numeric.is_finite():
                raise ValueError("timestamp must be finite")
            scale = {
                "nanosecond": Decimal("1e9"),
                "microsecond": Decimal("1e6"),
                "millisecond": Decimal("1e3"),
            }.get(unit or "", Decimal("1"))
            numeric /= scale
            result = datetime.fromtimestamp(float(numeric), tz=UTC)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("timestamp must not be empty")
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
                numeric = Decimal(text)
                if not numeric.is_finite():
                    raise ValueError("timestamp must be finite")
                inferred_unit = unit
                if inferred_unit is None:
                    absolute = abs(numeric)
                    if absolute >= Decimal("1e17"):
                        inferred_unit = "nanosecond"
                    elif absolute >= Decimal("1e14"):
                        inferred_unit = "microsecond"
                    elif absolute >= Decimal("1e11"):
                        inferred_unit = "millisecond"
                scale = {
                    "nanosecond": Decimal("1e9"),
                    "microsecond": Decimal("1e6"),
                    "millisecond": Decimal("1e3"),
                }.get(inferred_unit or "", Decimal("1"))
                numeric /= scale
                result = datetime.fromtimestamp(float(numeric), tz=UTC)
            else:
                iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
                result = datetime.fromisoformat(iso_text)
        else:
            raise ValueError(f"unsupported timestamp type: {type(value).__name__}")

    if result.tzinfo is None:
        return result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _find_time(mapping: Mapping[str, Any], candidates: Sequence[tuple[str, str | None]]) -> tuple[Any, str | None] | None:
    for key, unit in candidates:
        if key in mapping:
            return mapping[key], unit
    return None


def _format_validation_error(error: ValidationError) -> str:
    fragments: list[str] = []
    for item in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in item.get("loc", ())) or "span"
        fragments.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(fragments)


def _decode_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise ValueError("span event must be an object")
    payload = dict(event)
    payload.pop("droppedAttributesCount", None)
    payload.pop("dropped_attributes_count", None)
    attributes = payload.get("attributes", payload.get("attrs"))
    if attributes is not None:
        payload["attributes"] = decode_otlp_attributes(attributes)
    return payload


def _decode_link(link: Any) -> dict[str, Any]:
    if not isinstance(link, Mapping):
        raise ValueError("span link must be an object")
    payload = dict(link)
    payload.pop("droppedAttributesCount", None)
    payload.pop("dropped_attributes_count", None)
    for key in _TRACE_ID_KEYS:
        if key in payload:
            payload[key] = _decode_otlp_identifier(payload[key], width=32)
    for key in _SPAN_ID_KEYS:
        if key in payload:
            payload[key] = _decode_otlp_identifier(payload[key], width=16)
    attributes = payload.get("attributes", payload.get("attrs"))
    if attributes is not None:
        payload["attributes"] = decode_otlp_attributes(attributes)
    return payload


def span_from_mapping(
    mapping: Mapping[str, Any],
    *,
    trace_id: Any = None,
    parent_span_id: Any = None,
    inherited_attributes: Mapping[str, Any] | None = None,
    source: str = "<mapping>",
    record: int = 1,
) -> Span:
    """Convert one flat OpenInference/OTLP span mapping into a validated Span."""

    if not isinstance(mapping, Mapping):
        issue = ParseIssue(source, record, "invalid_span", "span record must be an object")
        raise SpanParseError(issue)

    payload = dict(mapping)
    context = payload.get("context")
    if isinstance(context, Mapping):
        if trace_id is None:
            trace_id = _first_present(context, _TRACE_ID_KEYS)
        if parent_span_id is None:
            parent_span_id = _first_present(context, _PARENT_ID_KEYS)

    selected_trace = _first_present(payload, _TRACE_ID_KEYS, trace_id)
    if selected_trace is None:
        issue = ParseIssue(source, record, "invalid_span", "trace_id is required")
        raise SpanParseError(issue)
    selected_parent = _first_present(payload, _PARENT_ID_KEYS, parent_span_id)
    selected_name = _first_present(payload, _NAME_KEYS)
    if selected_name is None:
        issue = ParseIssue(source, record, "invalid_span", "span name is required")
        raise SpanParseError(issue)

    model_payload: dict[str, Any] = {
        "trace_id": selected_trace,
        "span_id": _first_present(payload, _SPAN_ID_KEYS),
        "parent_span_id": selected_parent,
        "name": selected_name,
    }

    for target, key in (("kind", "kind"), ("span_kind", "spanKind")):
        if key in payload:
            model_payload["kind"] = payload[key]
            break

    start = _find_time(payload, _START_KEYS)
    if start is None:
        issue = ParseIssue(source, record, "invalid_span", "start time is required")
        raise SpanParseError(issue)
    model_payload["start_time"] = _coerce_datetime(start[0], start[1])

    end = _find_time(payload, _END_KEYS)
    if end is not None and end[0] not in (None, 0, "0", ""):
        model_payload["end_time"] = _coerce_datetime(end[0], end[1])

    status = payload.get("status")
    if isinstance(status, Mapping):
        code = _first_present(status, ("code", "status_code", "statusCode"))
        if code is not None:
            model_payload["status"] = code
        message = _first_present(
            status,
            ("message", "description", "status_message", "statusMessage"),
        )
        if message is not None:
            model_payload["status_message"] = message
    elif status is not None:
        model_payload["status"] = status

    for key in ("status_message", "statusMessage", "status_description", "statusDescription"):
        if key in payload:
            model_payload["status_message"] = payload[key]
            break

    raw_attributes = _first_present(payload, ("attributes", "attrs"), {})
    own_attributes = decode_otlp_attributes(raw_attributes)
    model_payload["attributes"] = _merge_attributes(inherited_attributes, own_attributes)

    if "events" in payload:
        events = payload["events"]
        if events is None:
            events = []
        if isinstance(events, Mapping):
            events = [events]
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
            issue = ParseIssue(source, record, "invalid_span", "events must be an array")
            raise SpanParseError(issue)
        model_payload["events"] = tuple(_decode_event(event) for event in events)

    if "links" in payload:
        links = payload["links"]
        if links is None:
            links = []
        if isinstance(links, Mapping):
            links = [links]
        if not isinstance(links, Sequence) or isinstance(links, (str, bytes, bytearray)):
            issue = ParseIssue(source, record, "invalid_span", "links must be an array")
            raise SpanParseError(issue)
        model_payload["links"] = tuple(_decode_link(link) for link in links)

    try:
        return Span.model_validate(model_payload)
    except ValidationError as exc:
        issue = ParseIssue(source, record, "invalid_span", _format_validation_error(exc))
        raise SpanParseError(issue) from exc
    except (TypeError, ValueError) as exc:
        issue = ParseIssue(source, record, "invalid_span", str(exc))
        raise SpanParseError(issue) from exc


def parse_span_document(document: Mapping[str, Any], **kwargs: Any) -> Span:
    """Parse a document containing exactly one span."""

    mappings = _extract_span_mappings(document)
    if len(mappings) != 1:
        source = str(kwargs.pop("source", "<document>"))
        record = int(kwargs.pop("record", 1))
        issue = ParseIssue(
            source,
            record,
            "invalid_document",
            f"expected exactly one span, found {len(mappings)}",
        )
        raise SpanParseError(issue)
    return span_from_mapping(mappings[0], source=str(kwargs.pop("source", "<document>")), record=int(kwargs.pop("record", 1)), **kwargs)


def _result_from_spans(
    spans: Sequence[Span],
    *,
    source_name: str,
    source_sha256: str,
    issues: Sequence[ParseIssue],
    source_format: TraceFormat,
) -> IngestionResult:
    if not spans:
        if issues:
            return IngestionResult(source_name, source_sha256, (), tuple(issues))
        raise EmptyTraceError(f"no spans found in '{source_name}'")

    grouped: dict[str, list[Span]] = {}
    for span in spans:
        grouped.setdefault(span.trace_id, []).append(span)

    traces = tuple(
        Trace(
            trace_id=trace_id,
            spans=tuple(group),
            source_name=source_name,
            source_sha256=source_sha256,
            source_format=source_format,
        )
        for trace_id, group in grouped.items()
    )
    return IngestionResult(source_name, source_sha256, traces, tuple(issues))


def _decode_text_documents(text: str) -> tuple[list[Any], list[tuple[int, str]]]:
    stripped = text.strip()
    if not stripped:
        return [], []

    if stripped[0] in "[{":
        try:
            return [_json_loads(stripped)], [(1, stripped)]
        except (json.JSONDecodeError, _DuplicateJSONKey, ValueError):
            pass

    documents: list[Any] = []
    decoded_lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        documents.append(_json_loads(line))
        decoded_lines.append((line_number, line))
    return documents, decoded_lines


class OpenInferenceSpanParser:
    """Strict-by-default direct parser for OpenInference and OTLP JSON."""

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    def parse(self, source: Any, *, source_name: str | None = None, strict: bool | None = None) -> IngestionResult:
        loaded = _load_source(source, source_name=source_name)
        fail_closed = self.strict if strict is None else strict
        if loaded.records is not None:
            return self.parse_records(
                loaded.records,
                source_name=loaded.name,
                source_sha256=loaded.sha256,
                strict=fail_closed,
            )
        assert loaded.text is not None
        return self._parse_text(
            loaded.text,
            source_name=loaded.name,
            source_sha256=loaded.sha256,
            strict=fail_closed,
        )

    def parse_file(self, path: str | os.PathLike[str], *, strict: bool | None = None) -> IngestionResult:
        return self.parse(Path(path), strict=strict)

    def parse_bytes(self, data: bytes | bytearray | memoryview, *, source_name: str = "<bytes>", strict: bool | None = None) -> IngestionResult:
        return self.parse(data, source_name=source_name, strict=strict)

    def parse_text(self, text: str, *, source_name: str = "<string>", strict: bool | None = None) -> IngestionResult:
        return self.parse(text, source_name=source_name, strict=strict)

    def parse_jsonl(self, source: Any, *, source_name: str | None = None, strict: bool | None = None) -> IngestionResult:
        return self.parse(source, source_name=source_name, strict=strict)

    def parse_document(self, document: Mapping[str, Any] | Sequence[Any], *, source_name: str = "<document>", strict: bool | None = None) -> IngestionResult:
        fail_closed = self.strict if strict is None else strict
        try:
            raw_hash = sha256_hex(document)
        except (TypeError, ValueError) as exc:
            issue = ParseIssue(source_name, 1, "invalid_document", str(exc))
            if fail_closed:
                raise SpanParseError(issue) from exc
            return IngestionResult(source_name, sha256_hex(repr(document)), (), (issue,))
        return self._parse_documents(
            [document],
            [(1, "")],
            source_name=source_name,
            source_sha256=raw_hash,
            strict=fail_closed,
            source_format=TraceFormat.OTLP_JSON if _is_otlp_document(document) else TraceFormat.OPENINFERENCE_JSON,
        )

    def parse_mapping(self, mapping: Mapping[str, Any], **kwargs: Any) -> IngestionResult:
        return self.parse_document(mapping, **kwargs)

    def parse_record(self, mapping: Mapping[str, Any], **kwargs: Any) -> Span:
        return span_from_mapping(mapping, **kwargs)

    def parse_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        source_name: str = "<records>",
        source_sha256: str | None = None,
        strict: bool | None = None,
    ) -> IngestionResult:
        fail_closed = self.strict if strict is None else strict
        typed_records = tuple(dict(record) for record in records)
        digest = source_sha256 or _hash_records(typed_records)
        return self._parse_documents(
            typed_records,
            [(index, "") for index in range(1, len(typed_records) + 1)],
            source_name=source_name,
            source_sha256=digest,
            strict=fail_closed,
            source_format=TraceFormat.OPENINFERENCE_JSONL,
        )

    def _parse_text(
        self,
        text: str,
        *,
        source_name: str,
        source_sha256: str,
        strict: bool,
    ) -> IngestionResult:
        try:
            documents, lines = _decode_text_documents(text)
        except json.JSONDecodeError as exc:
            issue = ParseIssue(source_name, exc.lineno, "invalid_json", f"record is not valid JSON: {exc.msg}")
            raise SpanParseError(issue) from exc
        except (_DuplicateJSONKey, ValueError) as exc:
            issue = ParseIssue(source_name, 1, "invalid_json", str(exc))
            raise SpanParseError(issue) from exc

        if not documents:
            raise EmptyTraceError(f"no JSON records found in '{source_name}'")

        source_format = (
            TraceFormat.OTLP_JSON
            if any(_is_otlp_document(document) for document in documents)
            else TraceFormat.OPENINFERENCE_JSONL
        )
        return self._parse_documents(
            documents,
            lines,
            source_name=source_name,
            source_sha256=source_sha256,
            strict=strict,
            source_format=source_format,
        )

    def _parse_documents(
        self,
        documents: Sequence[Any],
        line_information: Sequence[tuple[int, str]],
        *,
        source_name: str,
        source_sha256: str,
        strict: bool,
        source_format: TraceFormat,
    ) -> IngestionResult:
        spans: list[Span] = []
        by_identifier: dict[tuple[str, str], Span] = {}
        issues: list[ParseIssue] = []

        for index, document in enumerate(documents):
            record_number = line_information[index][0] if index < len(line_information) else index + 1
            try:
                mappings = _extract_span_mappings(document)
            except (TypeError, ValueError) as exc:
                issue = ParseIssue(source_name, record_number, "invalid_span", str(exc))
                if strict:
                    raise SpanParseError(issue) from exc
                issues.append(issue)
                continue

            for mapping in mappings:
                try:
                    span = span_from_mapping(mapping, source=source_name, record=record_number)
                except SpanParseError as exc:
                    if strict:
                        raise
                    issues.append(exc.issue)
                    continue

                key = (span.trace_id, span.span_id)
                previous = by_identifier.get(key)
                if previous is not None:
                    if previous.canonical_sha256 == span.canonical_sha256:
                        issue = ParseIssue(
                            source_name,
                            record_number,
                            "duplicate_span_ignored",
                            f"identical span '{span.span_id}' was already ingested",
                        )
                        issues.append(issue)
                        continue
                    issue = ParseIssue(
                        source_name,
                        record_number,
                        "conflicting_duplicate_span",
                        f"span '{span.span_id}' conflicts with an earlier canonical record",
                    )
                    raise DuplicateSpanError(issue)

                by_identifier[key] = span
                spans.append(span)

        return _result_from_spans(
            spans,
            source_name=source_name,
            source_sha256=source_sha256,
            issues=issues,
            source_format=source_format,
        )


class OpenTelemetrySpanAggregator:
    """Thread-safe incremental aggregator for OTLP resource/scope envelopes."""

    def __init__(self, *, strict: bool = True, source_name: str = "otel-aggregator"):
        self.strict = strict
        self.source_name = source_name
        self._spans: dict[tuple[str, str], Span] = {}
        self._source_hashes: list[str] = []
        self._issues: list[ParseIssue] = []
        self._lock = threading.RLock()

    @property
    def spans(self) -> tuple[Span, ...]:
        with self._lock:
            return tuple(self._spans.values())

    @property
    def trace_ids(self) -> tuple[str, ...]:
        with self._lock:
            result: list[str] = []
            for trace_id, _ in self._spans:
                if trace_id not in result:
                    result.append(trace_id)
            return tuple(result)

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()
            self._source_hashes.clear()
            self._issues.clear()

    def add_span(self, span: Span | Mapping[str, Any], *, source: str = "otel-aggregator") -> Span:
        if not isinstance(span, Span):
            span = span_from_mapping(span, source=source)
        with self._lock:
            key = (span.trace_id, span.span_id)
            existing = self._spans.get(key)
            if existing is not None:
                if existing.canonical_sha256 == span.canonical_sha256:
                    return existing
                issue = ParseIssue(
                    source,
                    1,
                    "conflicting_duplicate_span",
                    f"span '{span.span_id}' conflicts with previously aggregated evidence",
                )
                raise DuplicateSpanError(issue)
            self._spans[key] = span
            return span

    def add_document(
        self,
        document: Mapping[str, Any] | Sequence[Any],
        *,
        source: str | None = None,
        strict: bool | None = None,
    ) -> tuple[Span, ...]:
        fail_closed = self.strict if strict is None else strict
        source_name = source or self.source_name
        try:
            raw_hash = sha256_hex(document)
        except (TypeError, ValueError) as exc:
            issue = ParseIssue(source_name, 1, "invalid_document", str(exc))
            if fail_closed:
                raise SpanParseError(issue) from exc
            with self._lock:
                self._issues.append(issue)
            return ()

        try:
            mappings = _extract_span_mappings(document)
        except (TypeError, ValueError) as exc:
            issue = ParseIssue(source_name, 1, "invalid_span", str(exc))
            if fail_closed:
                raise SpanParseError(issue) from exc
            with self._lock:
                self._issues.append(issue)
            return ()

        added: list[Span] = []
        for mapping in mappings:
            try:
                before = len(self._spans)
                parsed = self.add_span(mapping, source=source_name)
                if len(self._spans) > before:
                    added.append(parsed)
            except SpanParseError as exc:
                if fail_closed:
                    raise
                with self._lock:
                    self._issues.append(exc.issue)

        with self._lock:
            self._source_hashes.append(raw_hash)
        return tuple(added)

    def add_otlp_document(self, document: Mapping[str, Any] | Sequence[Any], **kwargs: Any) -> tuple[Span, ...]:
        return self.add_document(document, **kwargs)

    def add_otlp_payload(self, document: Mapping[str, Any] | Sequence[Any], **kwargs: Any) -> tuple[Span, ...]:
        return self.add_document(document, **kwargs)

    def add_resource_spans(self, document: Mapping[str, Any] | Sequence[Any], **kwargs: Any) -> tuple[Span, ...]:
        return self.add_document(document, **kwargs)

    def ingest(self, document: Mapping[str, Any] | Sequence[Any], **kwargs: Any) -> tuple[Span, ...]:
        return self.add_document(document, **kwargs)

    def ingest_otlp(self, document: Mapping[str, Any] | Sequence[Any], **kwargs: Any) -> tuple[Span, ...]:
        return self.add_document(document, **kwargs)

    def ingest_jsonl(self, source: Any, *, source_name: str | None = None, strict: bool | None = None) -> IngestionResult:
        fail_closed = self.strict if strict is None else strict
        loaded = _load_source(source, source_name=source_name or self.source_name)
        if loaded.records is not None:
            for index, record in enumerate(loaded.records, start=1):
                self.add_document(record, source=loaded.name, strict=fail_closed)
            with self._lock:
                if loaded.sha256 not in self._source_hashes:
                    self._source_hashes.append(loaded.sha256)
        else:
            assert loaded.text is not None
            try:
                documents, _ = _decode_text_documents(loaded.text)
            except json.JSONDecodeError as exc:
                issue = ParseIssue(loaded.name, exc.lineno, "invalid_json", f"record is not valid JSON: {exc.msg}")
                if fail_closed:
                    raise SpanParseError(issue) from exc
                with self._lock:
                    self._issues.append(issue)
            else:
                for index, document in enumerate(documents, start=1):
                    self.add_document(document, source=loaded.name, strict=fail_closed)
                with self._lock:
                    self._source_hashes.append(loaded.sha256)
        return self.finalize(strict=fail_closed)

    def _source_digest(self) -> str:
        if not self._source_hashes:
            return sha256_hex(b"")
        if len(self._source_hashes) == 1:
            return self._source_hashes[0]
        return sha256_hex(tuple(self._source_hashes))

    def finalize(self, *, strict: bool | None = None) -> IngestionResult:
        fail_closed = self.strict if strict is None else strict
        with self._lock:
            issues = tuple(self._issues)
            spans = tuple(self._spans.values())
        if not spans:
            if issues and not fail_closed:
                return IngestionResult(self.source_name, self._source_digest(), (), issues)
            raise EmptyTraceError("the OpenTelemetry aggregator contains no spans")
        return _result_from_spans(
            spans,
            source_name=self.source_name,
            source_sha256=self._source_digest(),
            issues=issues,
            source_format=TraceFormat.OTLP_JSON,
        )

    def build_result(self, *, strict: bool | None = None) -> IngestionResult:
        return self.finalize(strict=strict)

    def result(self, *, strict: bool | None = None) -> IngestionResult:
        return self.finalize(strict=strict)

    @property
    def traces(self) -> tuple[Trace, ...]:
        return self.finalize().traces

    def __enter__(self) -> OpenTelemetrySpanAggregator:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.finalize(strict=False)


def load_spans(
    source: Any,
    *,
    source_name: str | None = None,
    strict: bool = True,
) -> IngestionResult:
    """Ingest JSON/JSONL and return all discovered traces with source integrity."""

    return OpenInferenceSpanParser().parse(source, source_name=source_name, strict=strict)


def ingest_jsonl(
    source: Any,
    *,
    source_name: str | None = None,
    strict: bool = True,
) -> IngestionResult:
    return load_spans(source, source_name=source_name, strict=strict)


def ingest_trace(
    source: Any,
    trace_id: str | None = None,
    *,
    source_name: str | None = None,
    strict: bool = True,
) -> Trace:
    result = load_spans(source, source_name=source_name, strict=strict)
    if trace_id is not None:
        return result.get_trace(trace_id)
    if len(result.traces) != 1:
        raise AmbiguousTraceError(result.trace_ids)
    return result.traces[0]


def parse_spans(
    source: Any,
    trace_id: str | None = None,
    *,
    source_name: str | None = None,
    strict: bool = True,
) -> Trace:
    return ingest_trace(source, trace_id, source_name=source_name, strict=strict)


def parse_spans_jsonl(
    source: Any,
    trace_id: str | None = None,
    *,
    source_name: str | None = None,
    strict: bool = True,
) -> Trace:
    return ingest_trace(source, trace_id, source_name=source_name, strict=strict)


def parse_jsonl(
    source: Any,
    *,
    source_name: str | None = None,
    strict: bool = True,
) -> tuple[Span, ...]:
    """Parse JSONL records into a flat tuple of validated spans."""

    return load_spans(source, source_name=source_name, strict=strict).spans


__all__ = [
    "AmbiguousTraceError",
    "DuplicateSpanError",
    "EmptyTraceError",
    "IngestionResult",
    "OpenInferenceSpanParser",
    "OpenTelemetrySpanAggregator",
    "ParseIssue",
    "SpanParseError",
    "TraceNotFoundError",
    "decode_otlp_attributes",
    "decode_otlp_value",
    "ingest_jsonl",
    "ingest_trace",
    "load_spans",
    "parse_jsonl",
    "parse_span_document",
    "parse_spans",
    "parse_spans_jsonl",
    "span_from_mapping",
]