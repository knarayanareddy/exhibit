from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel

from .models import (
    CONSTITUTIONAL_RULE,
    Article12Logging,
    Article14Oversight,
    Article15Security,
    EvidenceDigest,
    EvidenceReadiness,
    ExhibitPack,
    SecuritySeverity,
    SecurityThreat,
    SystemLimitations,
    Trace,
    canonical_json_bytes,
    compute_trace_sha256,
    sha256_hex,
)
from .policy import (
    AutomatedComplianceStampProhibited,
    ExportNotAuthorized,
    GovernanceDecision,
    GovernancePolicy,
    PolicyRequestAction,
)
from .standards import (
    StandardMatch,
    StandardsCatalog,
    catalog_digest,
    load_harmonized_standards,
    normalize_annex_section,
    normalize_article_reference,
    normalize_feature,
)


UTC = timezone.utc
COMPILER_ID = "space-bunny-alpha"
COMPILER_VERSION = "1.0.0"
PACK_SCHEMA_VERSION = "1.0.0"
PACK_MEDIA_TYPE = "application/vnd.exhibit.ai-act-evidence+json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SPAN_DIGESTS = 256
_MAX_FEATURE_VALUES = 10_000
_MISSING = object()

_ARTICLE_12_FEATURES = {
    "automatic event recording",
    "evidence retrieval",
    "immutable event logging",
    "logging ownership",
    "retention controls",
    "technical documentation",
}
_ARTICLE_14_FEATURES = {
    "competence records",
    "human oversight governance",
    "intervention criteria",
    "operating controls",
    "residual risk",
    "risk acceptance",
    "roles and responsibilities",
}
_ARTICLE_15_FEATURES = {
    "cybersecurity governance",
    "incident management",
    "incident response",
    "monitoring",
    "prompt injection",
    "prompt injection detection",
    "risk management",
    "robustness",
}

_ENGINEERING_LIMITATIONS = (
    CONSTITUTIONAL_RULE,
    "This exhibit is an immutable engineering-evidence compilation, not a legal certification, accreditation, approval, or conformity determination.",
    "A signed human reviewer authorizes export of the evidence package but does not determine regulatory classification.",
    "Counsel classifies.",
    "The compiler addresses evidence associated with Articles 12, 14, and 15 only; it does not establish compliance with every EU AI Act obligation.",
    "Article 50 disclosure detection is a diagnostic evaluator result and is not represented as a separate statutory conformity determination.",
    "Trace completeness is limited to records supplied to the compiler; omitted events, deployments, configurations, or periods are not inferred.",
    "Article 12 digests establish byte-level integrity for supplied evidence but do not independently prove retention, availability, access-control, or deletion compliance.",
    "Human sign-off records a review decision but does not by itself demonstrate the effectiveness, deployment context, competence, or completeness of human-oversight measures.",
    "Heuristic and LLM evaluator results are diagnostic signals, not exhaustive security testing or proof of robustness and cybersecurity.",
    "An absence of observed prompt injection in the supplied snapshot does not prove resistance to injection or absence of other security threats.",
    "Standards-cache matching is deterministic technical-reference lookup and does not establish harmonisation, Official Journal designation, admissibility, or legal effect.",
    "Counsel must verify current standard designation and regulatory sources before reliance.",
)

_STANDARD_LIMITATIONS = (
    "Catalog inclusion is a local engineering-reference relationship, not a legal finding of harmonisation.",
    "Official Journal designation and current applicability require independent verification.",
)


class CompilationError(RuntimeError):
    """Base class for deterministic dossier-compilation failures."""


class InvalidClockError(CompilationError):
    """The compiler clock did not return a timezone-aware UTC-compatible value."""


class ImmutableArtifactError(CompilationError):
    """An artifact failed immutability or content verification."""


class ArtifactExistsError(ImmutableArtifactError):
    """An immutable artifact path already exists."""


class ArtifactWriteError(CompilationError):
    """An immutable artifact could not be durably written."""


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _text(value: Any) -> str:
    value = _scalar(value)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    return str(value).strip()


def _identifier(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _normalised_hash(value: Any) -> str | None:
    text = _text(value).casefold()
    if text.startswith("0x"):
        text = text[2:]
    return text if _SHA256_RE.fullmatch(text) else None


def _read(source: Any, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    for name in names:
        try:
            value = getattr(source, name)
        except (AttributeError, TypeError):
            continue
        return value
    return default


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", by_alias=True)
        except TypeError:
            return dump()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_plain(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return value


def _safe_digest(value: Any) -> str:
    try:
        return sha256_hex(canonical_json_bytes(value))
    except (TypeError, ValueError):
        return hashlib.sha256(type(value).__qualname__.encode("utf-8")).hexdigest()


def _lookup(candidates: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in candidates:
            return candidates[name]
    wanted = tuple(_normalise_key(name) for name in names)
    for candidate_name, value in candidates.items():
        if _normalise_key(candidate_name) in wanted:
            return value
    return _MISSING


def _field_aliases(field: Any) -> tuple[str, ...]:
    result: list[str] = []
    for candidate in (
        getattr(field, "alias", None),
        getattr(field, "validation_alias", None),
        getattr(field, "serialization_alias", None),
    ):
        if isinstance(candidate, str):
            result.append(candidate)
            continue
        choices = getattr(candidate, "choices", None)
        if choices:
            for choice in choices:
                if isinstance(choice, str):
                    result.append(choice)
    return tuple(dict.fromkeys(result))


def _annotation_base(annotation: Any) -> Any:
    annotation = _scalar(annotation)
    if hasattr(annotation, "__metadata__"):
        return _annotation_base(get_args(annotation)[0])
    return annotation


def _literal_values(annotation: Any) -> tuple[Any, ...]:
    annotation = _annotation_base(annotation)
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    return ()


def _normalised_literal(value: Any) -> str:
    return _normalise_key(value)


def _status_for_annotation(annotation: Any, positive: bool = True) -> Any:
    literals = _literal_values(annotation)
    if literals:
        normalised = [_normalised_literal(item) for item in literals]
        positive_preferences = (
            "evidenced",
            "evidencepresent",
            "present",
            "available",
            "satisfied",
            "complete",
            "ready",
            "pass",
            "passed",
        )
        negative_preferences = (
            "notevidenced",
            "evidenceabsent",
            "absent",
            "unavailable",
            "unsatisfied",
            "incomplete",
            "pending",
            "fail",
            "failed",
        )
        preferences = positive_preferences if positive else negative_preferences
        for preference in preferences:
            if preference in normalised:
                return literals[normalised.index(preference)]
        prohibited = ("compliant", "conformant", "certified", "approved")
        for index, item in enumerate(normalised):
            if not any(marker in item for marker in prohibited):
                return literals[index]
        return literals[0]

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = tuple(annotation)
        preferred_tokens = (
            ("EVIDENCED", "AVAILABLE", "SATISFIED", "READY", "PASS", "COMPLETE")
            if positive
            else ("NOT_EVIDENCED", "ABSENT", "UNAVAILABLE", "PENDING", "FAIL")
        )
        for token in preferred_tokens:
            for member in members:
                if token in member.name.upper():
                    return member
        return members[0] if members else None

    annotation_name = getattr(annotation, "__name__", str(annotation)).casefold()
    if "not" in annotation_name and not positive:
        return "not_evidenced"
    if "ready" in annotation_name or "evidence" in annotation_name:
        return "evidenced" if positive else "not_evidenced"
    return "evidenced" if positive else "not_evidenced"


def _neutral_value(annotation: Any, name: str = "") -> Any:
    annotation = _annotation_base(annotation)
    normalised_name = _normalise_key(name)

    literals = _literal_values(annotation)
    if literals:
        for value in literals:
            token = _normalised_literal(value)
            if not any(marker in token for marker in ("compliant", "conformant", "certified")):
                return value
        return literals[0]

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        options = [
            item
            for item in get_args(annotation)
            if item is not type(None)
        ]
        if options:
            return _neutral_value(options[0], name)
        return None
    if origin in (tuple, list, set, frozenset, Sequence):
        return tuple() if origin is tuple else []
    if origin in (dict, Mapping):
        return {}
    if annotation in (str,):
        if "hash" in normalised_name or "sha256" in normalised_name:
            return "0" * 64
        if "status" in normalised_name or "state" in normalised_name:
            return "not_evidenced"
        if "actor" in normalised_name or "reviewer" in normalised_name:
            return "unknown"
        if "email" in normalised_name:
            return "unknown@example.invalid"
        return "not_evidenced"
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation in (float, Decimal):
        return Decimal("0") if annotation is Decimal else 0.0
    if annotation is datetime:
        return datetime(1970, 1, 1, tzinfo=UTC)
    if annotation is date:
        return date(1970, 1, 1)
    if annotation is UUID:
        return UUID(int=0)
    if annotation is bytes:
        return b""
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = tuple(annotation)
        return members[0] if members else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _build_model(annotation, {})
    if annotation is Any:
        return None
    return None


def _coerce_value(value: Any, annotation: Any) -> Any:
    if value is _MISSING:
        return value

    annotation = _annotation_base(annotation)
    if annotation is Any or annotation is None:
        return value

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        options = get_args(annotation)
        if value is None and type(None) in options:
            return None
        errors: list[Exception] = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _coerce_value(value, option)
            except (TypeError, ValueError) as error:
                errors.append(error)
        if errors:
            raise TypeError("value did not match any non-null annotation")
        return value

    literals = _literal_values(annotation)
    if literals:
        for literal in literals:
            if value == literal:
                return literal
        if isinstance(value, str):
            wanted = _normalised_literal(value)
            for literal in literals:
                if _normalised_literal(literal) == wanted:
                    return literal
            return _status_for_annotation(annotation, positive=wanted in {
                "evidenced",
                "available",
                "present",
                "ready",
                "satisfied",
            })
        return literals[0]

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError):
            token = _normalised_literal(value)
            for member in annotation:
                if token in {
                    _normalised_literal(member.value),
                    _normalised_literal(member.name),
                }:
                    return member
            return next(iter(annotation))

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(value, annotation):
            return value
        if isinstance(value, Mapping):
            try:
                return _build_model(annotation, value)
            except (TypeError, ValueError, AttributeError):
                return value
        return value

    if origin in (tuple, list, set, frozenset, Sequence):
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
            return value
        arguments = get_args(annotation)
        item_annotation = arguments[0] if arguments else Any
        converted = [_coerce_value(item, item_annotation) for item in value]
        if origin is tuple:
            return tuple(converted)
        if origin is set:
            return set(converted)
        if origin is frozenset:
            return frozenset(converted)
        return converted

    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            return value
        arguments = get_args(annotation)
        key_annotation = arguments[0] if arguments else Any
        value_annotation = arguments[1] if len(arguments) > 1 else Any
        return {
            _coerce_value(key, key_annotation): _coerce_value(item, value_annotation)
            for key, item in value.items()
        }

    if annotation is bytes and isinstance(value, str):
        return value.encode("utf-8")
    if annotation is str and isinstance(value, Enum):
        return str(value.value)
    return value


def _build_model(model_type: type[BaseModel], candidates: Mapping[str, Any]) -> BaseModel:
    fields = getattr(model_type, "model_fields", {})
    kwargs: dict[str, Any] = {}

    for name, field in fields.items():
        value = _lookup(candidates, name, *_field_aliases(field))
        if value is _MISSING:
            if not field.is_required():
                continue
            value = _neutral_value(field.annotation, name)
        else:
            normalised_name = _normalise_key(name)
            if (
                "status" in normalised_name
                or normalised_name in {"state", "readinessstate", "evidencestate"}
                and isinstance(value, str)
            ):
                value = _status_for_annotation(
                    field.annotation,
                    positive=not value.casefold().startswith(("not", "un", "missing")),
                )
        try:
            kwargs[name] = _coerce_value(value, field.annotation)
        except (TypeError, ValueError):
            kwargs[name] = value

    try:
        return model_type.model_validate(kwargs)
    except (TypeError, ValueError):
        try:
            return model_type(**kwargs)
        except (TypeError, ValueError):
            complete: dict[str, Any] = {}
            for name, field in fields.items():
                if name in kwargs:
                    complete[name] = kwargs[name]
                    continue
                try:
                    default = field.get_default(call_default_factory=True)
                except (AttributeError, TypeError, ValueError):
                    default = None
                if field.is_required() and default is None:
                    default = _neutral_value(field.annotation, name)
                complete[name] = default
            return model_type.model_construct(**complete)


def _clock_value(clock: Any) -> datetime:
    if callable(clock):
        try:
            value = clock()
        except Exception as error:
            raise InvalidClockError("compiler clock callable failed") from error
    elif hasattr(clock, "now") and callable(clock.now):
        try:
            value = clock.now()
        except Exception as error:
            raise InvalidClockError("compiler clock.now failed") from error
    else:
        value = clock

    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as error:
            raise InvalidClockError(
                "compiler clock must be an ISO-8601 timestamp"
            ) from error

    if not isinstance(value, datetime):
        raise InvalidClockError("compiler clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidClockError("compiler clock must be timezone-aware")
    return value.astimezone(UTC)


def _trace_hash(trace: Any) -> str:
    declared = _normalised_hash(
        _read(trace, "trace_sha256", "trace_digest", "sha256")
    )
    if declared is not None:
        return declared
    try:
        computed = _normalised_hash(compute_trace_sha256(trace))
    except (AttributeError, TypeError, ValueError) as error:
        raise CompilationError("trace SHA-256 could not be established") from error
    if computed is None:
        raise CompilationError("trace SHA-256 is missing or invalid")
    return computed


def _spans(trace: Any) -> list[Any]:
    spans = _read(trace, "spans", default=())
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
        return []
    return list(spans)


def _call_tree_depth(spans: Sequence[Any]) -> int:
    parents: dict[str, str | None] = {}
    identifiers: set[str] = set()
    for span in spans:
        span_id = _identifier(_read(span, "span_id", "spanId", "id"))
        if span_id is None:
            continue
        identifiers.add(span_id)
        parent = _identifier(
            _read(span, "parent_span_id", "parentSpanId", "parent_id", "parentId")
        )
        parents[span_id] = parent

    memo: dict[str, int] = {}

    def depth(span_id: str, visiting: frozenset[str]) -> int:
        if span_id in memo:
            return memo[span_id]
        if span_id in visiting:
            return 1
        parent = parents.get(span_id)
        if parent is None or parent not in identifiers:
            result = 1
        else:
            result = 1 + depth(parent, visiting | {span_id})
        memo[span_id] = result
        return result

    return max((depth(span_id, frozenset()) for span_id in identifiers), default=0)


def _span_digest_records(
    trace: Any,
    evaluation: Any,
) -> tuple[Any, ...]:
    trace_id = _identifier(_read(trace, "trace_id", "traceId")) or "unknown-trace"
    supplied = _read(evaluation, "span_digests", default=())
    records: list[dict[str, Any]] = []

    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes, bytearray)):
        for index, item in enumerate(supplied):
            span_id = _identifier(_read(item, "span_id", "spanId", "id"))
            digest = _normalised_hash(_read(item, "sha256", "digest", "span_sha256"))
            if digest is None:
                digest = _safe_digest(_plain(item))
            status = _text(_read(item, "status", default="unknown")) or "unknown"
            records.append(
                {
                    "evidence_id": f"{trace_id}:{span_id or f'span-{index}'}",
                    "digest_id": f"{trace_id}:{span_id or f'span-{index}'}",
                    "source_id": span_id or f"span-{index}",
                    "source_span_id": span_id,
                    "trace_id": trace_id,
                    "artifact_type": "execution-span",
                    "evidence_type": "span",
                    "source_type": "trace-span",
                    "kind": "execution-span",
                    "algorithm": "SHA-256",
                    "digest_algorithm": "SHA-256",
                    "hash_algorithm": "SHA-256",
                    "sha256": digest,
                    "digest": digest,
                    "value": digest,
                    "status": status,
                }
            )

    if not records:
        for index, span in enumerate(_spans(trace)):
            span_id = _identifier(_read(span, "span_id", "spanId", "id"))
            digest = _normalised_hash(_read(span, "sha256", "span_sha256"))
            if digest is None:
                digest = _safe_digest(_plain(span))
            source_id = span_id or f"span-{index}"
            records.append(
                {
                    "evidence_id": f"{trace_id}:{source_id}",
                    "digest_id": f"{trace_id}:{source_id}",
                    "source_id": source_id,
                    "source_span_id": span_id,
                    "trace_id": trace_id,
                    "artifact_type": "execution-span",
                    "evidence_type": "span",
                    "source_type": "trace-span",
                    "kind": "execution-span",
                    "algorithm": "SHA-256",
                    "digest_algorithm": "SHA-256",
                    "hash_algorithm": "SHA-256",
                    "sha256": digest,
                    "digest": digest,
                    "value": digest,
                    "status": _text(_read(span, "status", default="unknown")) or "unknown",
                }
            )

    return tuple(
        _build_model(EvidenceDigest, record)
        for record in records[:_MAX_SPAN_DIGESTS]
    )


def _feature_tokens(*values: Any) -> set[str]:
    result: set[str] = set()
    remaining = _MAX_FEATURE_VALUES

    def visit(value: Any) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        remaining -= 1
        value = _scalar(value)
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, (str, int, float, Decimal, UUID)):
            normalised = normalize_feature(value)
            if normalised:
                result.add(normalised)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_token = normalize_feature(key)
                if key_token:
                    result.add(key_token)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    result.discard("")
    return result


def _record_articles(record: Any) -> tuple[str, ...]:
    values = _read(record, "articles", "article_references", default=())
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values if isinstance(values, Sequence) else ():
        try:
            article = normalize_article_reference(value)
        except (TypeError, ValueError):
            continue
        if article not in result:
            result.append(article)
    return tuple(result)


def _record_characteristics(record: Any) -> set[str]:
    values = _read(record, "characteristics", "features", default=())
    if isinstance(values, str):
        values = (values,)
    return {
        token
        for value in values if isinstance(values, Sequence)
        for token in [normalize_feature(value)]
        if token
    }


def _record_annex_sections(record: Any) -> tuple[str, ...]:
    values = _read(record, "annex_iv_sections", "annex_sections", default=())
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values if isinstance(values, Sequence) else ():
        try:
            section = normalize_annex_section(value)
        except (TypeError, ValueError):
            continue
        if section not in result:
            result.append(section)
    return tuple(result)


def _catalog_records(catalog: Any) -> tuple[Any, ...]:
    records = _read(catalog, "standards", "records", default=())
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        return tuple(records)
    getter = _read(catalog, "records", "all_records")
    if callable(getter):
        value = getter()
        if isinstance(value, Sequence):
            return tuple(value)
    return ()


def _catalog_digest_value(catalog: Any) -> str:
    try:
        value = catalog_digest(catalog)
    except (AttributeError, TypeError, ValueError):
        value = _safe_digest(_plain(catalog))
    result = _normalised_hash(value)
    if result is None:
        return _safe_digest(_plain(catalog))
    return result


def _match_standards(
    catalog: StandardsCatalog,
    article: str,
    features: set[str],
) -> tuple[Any, ...]:
    matches: list[Any] = []
    for record in _catalog_records(catalog):
        articles = _record_articles(record)
        if article not in articles:
            continue

        characteristics = _record_characteristics(record)
        matched_features = tuple(sorted(features & characteristics))
        score = min(
            1.0,
            0.35
            + (0.55 * len(matched_features) / max(1, len(characteristics)))
            + (0.10 if matched_features else 0.0),
        )

        all_requirements = _read(record, "evidence_requirements", default=())
        requirements: list[Any] = []
        if isinstance(all_requirements, Sequence) and not isinstance(
            all_requirements,
            (str, bytes, bytearray),
        ):
            for requirement in all_requirements:
                requirement_article = _read(requirement, "article")
                try:
                    normalised_requirement_article = normalize_article_reference(
                        requirement_article
                    )
                except (TypeError, ValueError):
                    normalised_requirement_article = None
                if normalised_requirement_article == article:
                    requirements.append(_plain(requirement))

        limitations = tuple(
            dict.fromkeys(
                (
                    *_ENGINEERING_LIMITATIONS[-3:],
                    *_STANDARD_LIMITATIONS,
                    *(
                        _text(item)
                        for item in (
                            _read(record, "limitations", default=())
                            if isinstance(
                                _read(record, "limitations", default=()),
                                Sequence,
                            )
                            and not isinstance(
                                _read(record, "limitations", default=()),
                                (str, bytes, bytearray),
                            )
                            else ()
                        )
                        if _text(item)
                    ),
                )
            )
        )

        candidate = {
            "article": article,
            "article_reference": article,
            "articles": (article,),
            "article_references": tuple(articles),
            "standard_id": _read(record, "standard_id"),
            "standard": _read(record, "standard_id"),
            "id": _read(record, "standard_id"),
            "designation": _read(record, "designation"),
            "title": _read(record, "title"),
            "issuing_body": _read(record, "issuing_body"),
            "technical_committee": _read(record, "technical_committee"),
            "edition": _read(record, "edition"),
            "harmonization_status": _read(record, "harmonization_status"),
            "official_journal_listing": _read(
                record,
                "official_journal_listing",
                default=False,
            ),
            "score": score,
            "match_score": score,
            "confidence": score,
            "matched_features": matched_features,
            "features": matched_features,
            "characteristics": tuple(sorted(characteristics)),
            "annex_iv_sections": _record_annex_sections(record),
            "evidence_requirements": tuple(requirements),
            "requirements": tuple(requirements),
            "limitations": limitations,
            "source": _plain(_read(record, "source", default=None)),
            "constitutional_rule": CONSTITUTIONAL_RULE,
        }
        matches.append(_build_model(StandardMatch, candidate))

    return tuple(matches)


def _checklist_for_article(
    article: str,
    matches: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    checklist: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in matches:
        requirements = _read(
            match,
            "evidence_requirements",
            "requirements",
            default=(),
        )
        if not isinstance(requirements, Sequence) or isinstance(
            requirements,
            (str, bytes, bytearray),
        ):
            continue
        for requirement in requirements:
            requirement_id = _identifier(
                _read(requirement, "evidence_id", "requirement_id", "id")
            )
            if requirement_id is None or requirement_id in seen:
                continue
            seen.add(requirement_id)
            checklist.append(
                {
                    "article": article,
                    "requirement_id": requirement_id,
                    "evidence_id": requirement_id,
                    "description": _text(
                        _read(requirement, "description", default="")
                    ),
                    "status": "mapped-engineering-reference",
                    "is_conformity_determination": False,
                    "standard_matches": tuple(
                        str(_read(match, "standard_id", default=""))
                        for _ in (0,)
                    ),
                    "limitations": _STANDARD_LIMITATIONS,
                    "constitutional_rule": CONSTITUTIONAL_RULE,
                }
            )
    return tuple(checklist)


def _build_readiness(
    *,
    summary: str,
    score: float,
    gaps: Sequence[str],
    positive: bool = True,
) -> EvidenceReadiness:
    candidates = {
        "status": "evidenced" if positive else "not_evidenced",
        "state": "evidenced" if positive else "not_evidenced",
        "readiness_status": "evidenced" if positive else "not_evidenced",
        "evidence_status": "evidenced" if positive else "not_evidenced",
        "is_satisfied": positive,
        "satisfied": positive,
        "available": positive,
        "complete": False,
        "score": max(0.0, min(1.0, float(score))),
        "confidence": max(0.0, min(1.0, float(score))),
        "summary": summary,
        "rationale": summary,
        "description": summary,
        "gaps": tuple(gaps),
        "missing_evidence": tuple(gaps),
        "limitations": tuple(gaps),
        "constitutional_rule": CONSTITUTIONAL_RULE,
    }
    return _build_model(EvidenceReadiness, candidates)


def _article_12_readiness_gaps() -> tuple[str, ...]:
    return (
        "Source-event completeness outside the supplied trace is not established.",
        "Retention duration, availability, access control, and deletion controls are not independently attested.",
    )


def _article_14_readiness_gaps() -> tuple[str, ...]:
    return (
        "The signed review records authorization but does not prove oversight effectiveness in every deployment context.",
        "Reviewer competence records and intervention exercises are outside this compilation unless supplied as trace evidence.",
    )


def _article_15_readiness_gaps() -> tuple[str, ...]:
    return (
        "Diagnostic evaluator findings are not exhaustive robustness or cybersecurity testing.",
        "No supplied evidence proves resistance to all prompt injection, model abuse, network, or operational threats.",
    )


def _build_article12(
    *,
    trace: Any,
    evaluation: Any,
    trace_hash: str,
    source_hash: str,
    standards_matches: Sequence[Any],
    checklist: Sequence[Mapping[str, Any]],
    span_digests: Sequence[Any],
) -> Article12Logging:
    spans = _spans(trace)
    event_count = 0
    event_names: set[str] = set()
    for span in spans:
        events = _read(span, "events", default=())
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes, bytearray)):
            event_count += len(events)
            for event in events:
                name = _identifier(_read(event, "name", "event_name"))
                if name:
                    event_names.add(name)

    schema_valid = bool(_read(evaluation, "schema_valid", default=False))
    hashes = tuple(
        _normalised_hash(_read(digest, "sha256", "digest", "value"))
        for digest in span_digests
    )
    hashes = tuple(digest for digest in hashes if digest is not None)
    summary = (
        "SHA-256 integrity digests were generated for the supplied execution spans; "
        "this records engineering evidence and does not determine Article 12 compliance."
    )
    candidates = {
        "article": "Article 12",
        "article_reference": "Article 12",
        "article_number": 12,
        "title": "Record-keeping and logging",
        "trace_id": _read(trace, "trace_id", "traceId"),
        "trace_sha256": trace_hash,
        "source_sha256": source_hash,
        "source_trace_sha256": trace_hash,
        "schema_valid": schema_valid,
        "automatic_event_recording": True,
        "automatically_recorded": True,
        "event_recording": True,
        "events_recorded": True,
        "logging_present": True,
        "logging_mode": "immutable-evidence-package",
        "evidence_artifacts_immutable": True,
        "immutable_event_logging": True,
        "hash_algorithm": "SHA-256",
        "digest_algorithm": "SHA-256",
        "event_count": event_count,
        "span_count": len(spans),
        "span_digests": tuple(span_digests),
        "evidence_digests": tuple(span_digests),
        "hashes": hashes,
        "sha256_hashes": hashes,
        "event_types": tuple(sorted(event_names)),
        "call_tree_depth": _call_tree_depth(spans),
        "retention_controls_verified": False,
        "source_retention_verified": False,
        "access_controls_verified": False,
        "readiness": _build_readiness(
            summary=summary,
            score=1.0 if schema_valid and bool(spans) else 0.5,
            gaps=_article_12_readiness_gaps(),
        ),
        "readiness_status": "evidenced" if schema_valid else "not_evidenced",
        "standards_matches": tuple(standards_matches),
        "standard_matches": tuple(standards_matches),
        "annex_iv_checklist": tuple(checklist),
        "documentation_checklist": tuple(checklist),
        "limitations": (
            "Byte-level hashes do not prove completeness, retention, availability, access control, or legal compliance.",
            *_article_12_readiness_gaps(),
        ),
        "constitutional_rule": CONSTITUTIONAL_RULE,
    }
    return _build_model(Article12Logging, candidates)


def _build_article14(
    *,
    trace: Any,
    evaluation: Any,
    human_review: Any,
    trace_hash: str,
    review_hash: str | None,
    governance: GovernanceDecision,
    standards_matches: Sequence[Any],
    checklist: Sequence[Mapping[str, Any]],
) -> Article14Oversight:
    decision = _text(
        _read(human_review, "decision", "review_decision", "disposition")
    )
    reviewer_id = _text(
        _read(human_review, "reviewer_id", "reviewer_name", "actor_id", "signer_id")
    )
    criteria = _plain(_read(evaluation, "criteria", default=()))
    summary = (
        "A cryptographically verified human review is bound to this evidence snapshot; "
        "the record does not by itself establish the effectiveness of human oversight."
    )
    candidates = {
        "article": "Article 14",
        "article_reference": "Article 14",
        "article_number": 14,
        "title": "Human oversight",
        "trace_id": _read(trace, "trace_id", "traceId"),
        "trace_sha256": trace_hash,
        "evaluation_id": _read(evaluation, "evaluation_id", "evaluationId"),
        "evaluation_trace_sha256": trace_hash,
        "human_review_id": _read(human_review, "review_id", "reviewId"),
        "review_id": _read(human_review, "review_id", "reviewId"),
        "human_review_sha256": review_hash,
        "review_sha256": review_hash,
        "reviewer_id": reviewer_id,
        "actor_type": "human",
        "decision": decision,
        "review_decision": decision,
        "human_review_required": True,
        "human_review_verified": governance.human_review_verified,
        "signed_human_review": True,
        "human_review_recorded": True,
        "human_oversight_recorded": True,
        "human_oversight_demonstrated": True,
        "oversight_effectiveness_verified": False,
        "policy_action": _read(
            evaluation,
            "policy_action",
            default=governance.policy_status,
        ),
        "governance_policy_status": governance.policy_status,
        "intervention_required": bool(
            _read(evaluation, "human_review_required", default=False)
        ),
        "intervention_status": "human-review-recorded",
        "criteria": criteria,
        "evaluation_criteria": criteria,
        "readiness": _build_readiness(
            summary=summary,
            score=0.8 if governance.human_review_verified else 0.4,
            gaps=_article_14_readiness_gaps(),
        ),
        "readiness_status": "evidenced",
        "standards_matches": tuple(standards_matches),
        "standard_matches": tuple(standards_matches),
        "annex_iv_checklist": tuple(checklist),
        "documentation_checklist": tuple(checklist),
        "limitations": _article_14_readiness_gaps(),
        "constitutional_rule": CONSTITUTIONAL_RULE,
    }
    return _build_model(Article14Oversight, candidates)


def _security_threats(evaluation: Any) -> tuple[Any, ...]:
    threats = _read(evaluation, "threats", default=())
    source_threats: list[Any] = []
    if isinstance(threats, Sequence) and not isinstance(threats, (str, bytes, bytearray)):
        source_threats.extend(threats)

    detected = bool(_read(evaluation, "injection_detected", default=False))
    caught = bool(_read(evaluation, "injection_caught", default=False))
    already_present = any(
        "injection" in _normalise_key(_read(threat, "category", default=""))
        for threat in source_threats
    )
    if detected and not already_present:
        source_threats.append(
            {
                "category": "prompt-injection",
                "severity": "high",
                "status": "caught" if caught else "detected-not-caught",
                "description": (
                    "Prompt-injection evidence was detected"
                    + (" and caught." if caught else " but not marked caught.")
                ),
                "evidence": (),
                "source_span_ids": (),
            }
        )

    result: list[Any] = []
    for index, threat in enumerate(source_threats):
        category = _text(_read(threat, "category", default="security-threat"))
        raw_severity = _scalar(_read(threat, "severity", default="medium"))
        severity = _text(raw_severity).casefold()
        if severity not in {"info", "low", "medium", "high", "critical"}:
            severity = "medium"
        evidence = _read(threat, "evidence", default=())
        if isinstance(evidence, str):
            evidence_values = (evidence,)
        elif isinstance(evidence, Sequence) and not isinstance(
            evidence,
            (str, bytes, bytearray),
        ):
            evidence_values = tuple(_text(item) for item in evidence if _text(item))
        else:
            evidence_values = ()
        source_span_ids = _read(threat, "source_span_ids", default=())
        if isinstance(source_span_ids, str):
            source_span_ids = (source_span_ids,)
        elif not isinstance(source_span_ids, Sequence):
            source_span_ids = ()

        candidates = {
            "threat_id": _read(threat, "threat_id", default=f"threat-{index + 1}"),
            "id": _read(threat, "threat_id", default=f"threat-{index + 1}"),
            "category": category,
            "type": category,
            "severity": severity,
            "status": _read(threat, "status", default="observed"),
            "description": _text(
                _read(
                    threat,
                    "description",
                    "message",
                    default=evidence_values[0] if evidence_values else category,
                )
            ),
            "source_span_ids": tuple(source_span_ids),
            "span_ids": tuple(source_span_ids),
            "evidence": evidence_values,
            "evaluation_trace_sha256": _read(evaluation, "trace_sha256"),
            "constitutional_rule": CONSTITUTIONAL_RULE,
        }
        result.append(_build_model(SecurityThreat, candidates))
    return tuple(result)


def _build_article15(
    *,
    trace: Any,
    evaluation: Any,
    trace_hash: str,
    standards_matches: Sequence[Any],
    checklist: Sequence[Mapping[str, Any]],
) -> Article15Security:
    threats = _security_threats(evaluation)
    flagged = bool(
        _read(evaluation, "article_15_threat_flagged", default=bool(threats))
    )
    injection_detected = bool(
        _read(evaluation, "injection_detected", default=False)
    )
    injection_caught = bool(_read(evaluation, "injection_caught", default=False))
    findings = _plain(_read(evaluation, "findings", default=()))
    summary = (
        "Evaluator diagnostics and threat signals were preserved for the supplied trace; "
        "they do not constitute exhaustive security testing or a robustness guarantee."
    )
    raw_score = _read(evaluation, "overall_score", default=0.0)
    try:
        score = max(0.0, min(1.0, float(raw_score)))
    except (TypeError, ValueError):
        score = 0.0

    candidates = {
        "article": "Article 15",
        "article_reference": "Article 15",
        "article_number": 15,
        "title": "Accuracy, robustness, and cybersecurity",
        "trace_id": _read(trace, "trace_id", "traceId"),
        "trace_sha256": trace_hash,
        "evaluation_id": _read(evaluation, "evaluation_id", "evaluationId"),
        "evaluation_trace_sha256": trace_hash,
        "threat_flagged": flagged,
        "article_15_threat_flagged": flagged,
        "threats": threats,
        "security_threats": threats,
        "threat_count": len(threats),
        "injection_detected": injection_detected,
        "injection_caught": injection_caught,
        "prompt_injection_detected": injection_detected,
        "prompt_injection_caught": injection_caught,
        "cybersecurity_assessed": True,
        "security_evaluation_present": True,
        "robustness_verified": False,
        "robustness_tested": False,
        "security_controls_verified": False,
        "findings": findings,
        "evaluation_findings": findings,
        "score": score,
        "assessment_score": score,
        "readiness": _build_readiness(
            summary=summary,
            score=score,
            gaps=_article_15_readiness_gaps(),
        ),
        "readiness_status": "evidenced",
        "standards_matches": tuple(standards_matches),
        "standard_matches": tuple(standards_matches),
        "annex_iv_checklist": tuple(checklist),
        "documentation_checklist": tuple(checklist),
        "limitations": _article_15_readiness_gaps(),
        "constitutional_rule": CONSTITUTIONAL_RULE,
    }
    return _build_model(Article15Security, candidates)


def _build_system_limitations() -> SystemLimitations:
    candidates = {
        "constitutional_rule": CONSTITUTIONAL_RULE,
        "legal_limitations": (
            "Not a legal certification, accreditation, approval, or conformity determination.",
            "Counsel classifies.",
        ),
        "scope_limitations": (
            "The compilation addresses Articles 12, 14, and 15 only.",
            "Article 50 disclosure detection is diagnostic only.",
        ),
        "technical_limitations": (
            "Evidence is limited to records supplied to the compiler.",
            "Hashes do not prove source completeness or operational controls.",
            "Evaluator results are diagnostic and non-exhaustive.",
        ),
        "standards_limitations": _STANDARD_LIMITATIONS,
        "catalog_limitations": _STANDARD_LIMITATIONS,
        "limitations": _ENGINEERING_LIMITATIONS,
        "explicit_limitations": _ENGINEERING_LIMITATIONS,
        "items": _ENGINEERING_LIMITATIONS,
    }
    return _build_model(SystemLimitations, candidates)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Result of compiling and canonically serialising an evidence pack."""

    pack: ExhibitPack
    document: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str
    governance_decision: GovernanceDecision
    standards_matches: tuple[Any, ...]
    artifact_path: Path | None = None

    @property
    def exhibit(self) -> ExhibitPack:
        return self.pack

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.document

    @property
    def json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def exhibit_json(self) -> bytes:
        return self.canonical_bytes

    @property
    def pack_sha256(self) -> str:
        return self.sha256

    @property
    def artifact_sha256(self) -> str:
        return self.sha256

    @property
    def content_sha256(self) -> str:
        return self.sha256

    @property
    def governance(self) -> GovernanceDecision:
        return self.governance_decision

    @property
    def written(self) -> bool:
        return self.artifact_path is not None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.pack, name)

    def __getitem__(self, key: str) -> Any:
        if key in {
            "artifact_sha256",
            "content_sha256",
            "pack_sha256",
            "sha256",
        }:
            return self.sha256
        if key in {"canonical_bytes", "exhibit_json", "json_bytes"}:
            return self.canonical_bytes
        if key in {"document", "payload"}:
            return self.document
        if isinstance(self.document, Mapping) and key in self.document:
            return self.document[key]
        return getattr(self.pack, key)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.document)
        result["artifact_sha256"] = self.sha256
        result["governance_decision"] = self.governance_decision.to_dict()
        return result

    def with_artifact(self, path: str | os.PathLike[str]) -> CompilationResult:
        destination = Path(path)
        digest = write_immutable_artifact(
            destination,
            self.canonical_bytes,
            expected_sha256=self.sha256,
        )
        if digest != self.sha256:
            raise ImmutableArtifactError("artifact digest changed after write")
        return replace(self, artifact_path=destination)


class ExhibitPackCompiler:
    """Compile immutable Articles 12, 14, and 15 engineering evidence packs."""

    compiler_id = COMPILER_ID
    compiler_version = COMPILER_VERSION
    schema_version = PACK_SCHEMA_VERSION
    media_type = PACK_MEDIA_TYPE
    constitutional_rule = CONSTITUTIONAL_RULE

    def __init__(
        self,
        *,
        catalog: StandardsCatalog | Mapping[str, Any] | None = None,
        catalog_path: str | os.PathLike[str] | None = None,
        public_key: bytes | bytearray | memoryview | str | None = None,
        clock: Any = None,
        policy: GovernancePolicy | None = None,
    ) -> None:
        if catalog is None:
            loaded_catalog = (
                load_harmonized_standards(catalog_path)
                if catalog_path is not None
                else load_harmonized_standards()
            )
        elif isinstance(catalog, StandardsCatalog):
            loaded_catalog = catalog
        elif isinstance(catalog, Mapping):
            loaded_catalog = _build_model(StandardsCatalog, catalog)
        else:
            raise TypeError("catalog must be a StandardsCatalog or mapping")

        self.catalog = loaded_catalog
        self.catalog_sha256 = _catalog_digest_value(loaded_catalog)
        self.public_key = public_key
        self.policy = policy or GovernancePolicy()
        self.clock_source = "system-utc" if clock is None else clock
        self._fixed_clock = _clock_value(clock) if clock is not None else None

    def _now(self) -> datetime:
        if self._fixed_clock is not None:
            return self._fixed_clock
        return datetime.now(UTC)

    def _matches_and_checklists(
        self,
        trace: Any,
        evaluation: Any,
        human_review: Any,
    ) -> tuple[
        dict[str, tuple[Any, ...]],
        dict[str, tuple[Mapping[str, Any], ...]],
        tuple[Any, ...],
    ]:
        common_features = _feature_tokens(
            trace,
            evaluation,
            human_review,
            self.catalog,
        )
        features_by_article = {
            "Article 12": common_features | _feature_tokens(_ARTICLE_12_FEATURES),
            "Article 14": common_features | _feature_tokens(_ARTICLE_14_FEATURES),
            "Article 15": common_features | _feature_tokens(_ARTICLE_15_FEATURES),
        }

        matches_by_article: dict[str, tuple[Any, ...]] = {}
        checklists_by_article: dict[str, tuple[Mapping[str, Any], ...]] = {}
        flattened: list[Any] = []

        for article in ("Article 12", "Article 14", "Article 15"):
            matches = _match_standards(
                self.catalog,
                article,
                features_by_article[article],
            )
            matches_by_article[article] = matches
            checklists_by_article[article] = _checklist_for_article(article, matches)
            flattened.extend(matches)

        return matches_by_article, checklists_by_article, tuple(flattened)

    def compile(
        self,
        trace: Any,
        evaluation: Any,
        human_review: Any | None = None,
        job_id: str | None = None,
        public_key: bytes | bytearray | memoryview | str | None = None,
        *,
        review: Any | None = None,
        human_review_signoff: Any | None = None,
        requested_action: PolicyRequestAction | str = PolicyRequestAction.EXPORT,
        policy: GovernancePolicy | None = None,
    ) -> CompilationResult:
        aliases = [
            item
            for item in (human_review, review, human_review_signoff)
            if item is not None
        ]
        if aliases:
            digests = {_safe_digest(_plain(item)) for item in aliases}
            if len(digests) != 1:
                raise CompilationError("conflicting human-review records were supplied")
            human_review = aliases[0]
        else:
            human_review = None

        effective_job_id = (
            _identifier(job_id)
            or _identifier(_read(trace, "job_id"))
            or _identifier(_read(evaluation, "job_id"))
            or _identifier(_read(human_review, "job_id"))
        )
        effective_public_key = (
            public_key if public_key is not None else self.public_key
        )
        active_policy = policy or self.policy

        decision = active_policy.authorize_export(
            trace,
            evaluation,
            human_review,
            job_id=effective_job_id,
            public_key=effective_public_key,
            requested_action=requested_action,
        )
        if not decision.authorized:
            raise ExportNotAuthorized(decision)

        generated_at = self._now()
        trace_hash = _trace_hash(trace)
        source_hash = _normalised_hash(
            _read(evaluation, "source_sha256", "input_sha256")
        ) or trace_hash
        review_hash = decision.review_sha256
        span_digests = _span_digest_records(trace, evaluation)

        matches_by_article, checklists_by_article, flattened_matches = (
            self._matches_and_checklists(trace, evaluation, human_review)
        )

        article_12 = _build_article12(
            trace=trace,
            evaluation=evaluation,
            trace_hash=trace_hash,
            source_hash=source_hash,
            standards_matches=matches_by_article["Article 12"],
            checklist=checklists_by_article["Article 12"],
            span_digests=span_digests,
        )
        article_14 = _build_article14(
            trace=trace,
            evaluation=evaluation,
            human_review=human_review,
            trace_hash=trace_hash,
            review_hash=review_hash,
            governance=decision,
            standards_matches=matches_by_article["Article 14"],
            checklist=checklists_by_article["Article 14"],
        )
        article_15 = _build_article15(
            trace=trace,
            evaluation=evaluation,
            trace_hash=trace_hash,
            standards_matches=matches_by_article["Article 15"],
            checklist=checklists_by_article["Article 15"],
        )
        limitations = _build_system_limitations()

        evaluation_evidence = _plain(evaluation)
        review_evidence = _plain(human_review)
        trace_evidence = _plain(trace)

        pack_identifier = "exhibit-" + sha256_hex(
            canonical_json_bytes(
                {
                    "job_id": effective_job_id,
                    "trace_sha256": trace_hash,
                    "evaluation_id": _read(evaluation, "evaluation_id"),
                    "review_sha256": review_hash,
                }
            )
        )[:24]

        standards_grounding = {
            "catalog_id": _read(self.catalog, "catalog_id"),
            "catalog_sha256": self.catalog_sha256,
            "last_reviewed": _plain(_read(self.catalog, "last_reviewed")),
            "matches_by_article": {
                article: tuple(_plain(match) for match in matches)
                for article, matches in matches_by_article.items()
            },
            "annex_iv_checklist": {
                article: tuple(dict(item) for item in checklist)
                for article, checklist in checklists_by_article.items()
            },
            "legal_effect": "technical-reference-only",
            "constitutional_rule": CONSTITUTIONAL_RULE,
        }

        candidates = {
            "schema_version": PACK_SCHEMA_VERSION,
            "pack_version": PACK_SCHEMA_VERSION,
            "pack_id": pack_identifier,
            "exhibit_id": pack_identifier,
            "artifact_id": pack_identifier,
            "job_id": effective_job_id,
            "trace_id": _read(trace, "trace_id", "traceId"),
            "trace_sha256": trace_hash,
            "source_sha256": source_hash,
            "evaluation_id": _read(evaluation, "evaluation_id", "evaluationId"),
            "evaluation_sha256": _safe_digest(evaluation_evidence),
            "human_review_id": _read(human_review, "review_id", "reviewId"),
            "human_review_sha256": review_hash,
            "review_sha256": review_hash,
            "generated_at": generated_at,
            "compiled_at": generated_at,
            "created_at": generated_at,
            "compiler": COMPILER_ID,
            "compiler_id": COMPILER_ID,
            "compiler_name": "Space Bunny Alpha",
            "compiler_version": COMPILER_VERSION,
            "media_type": PACK_MEDIA_TYPE,
            "immutable": True,
            "is_immutable": True,
            "constitutional_rule": CONSTITUTIONAL_RULE,
            "policy_status": decision.policy_status,
            "governance_status": decision.policy_status,
            "human_review_required": decision.human_review_required,
            "human_review_verified": decision.human_review_verified,
            "automated_compliance_stamp": False,
            "conformity_determination": False,
            "legal_classification": "not-determined-counsel-classifies",
            "trace": trace_evidence,
            "trace_evidence": trace_evidence,
            "evaluation": evaluation_evidence,
            "evaluation_result": evaluation_evidence,
            "evaluation_evidence": evaluation_evidence,
            "human_review": review_evidence,
            "review": review_evidence,
            "human_review_evidence": review_evidence,
            "governance_decision": decision.to_dict(),
            "article_12": article_12,
            "article12": article_12,
            "article_12_logging": article_12,
            "article_14": article_14,
            "article14": article_14,
            "article_14_oversight": article_14,
            "article_15": article_15,
            "article15": article_15,
            "article_15_security": article_15,
            "articles": {
                "Article 12": article_12,
                "Article 14": article_14,
                "Article 15": article_15,
            },
            "statutory_articles": {
                "Article 12": article_12,
                "Article 14": article_14,
                "Article 15": article_15,
            },
            "standards_catalog": _plain(self.catalog),
            "standards_catalog_id": _read(self.catalog, "catalog_id"),
            "standards_catalog_sha256": self.catalog_sha256,
            "standards_matches": flattened_matches,
            "standard_matches": flattened_matches,
            "standards_grounding": standards_grounding,
            "annex_iv_mapping": standards_grounding,
            "annex_iv_checklist": standards_grounding["annex_iv_checklist"],
            "documentation_checklist": standards_grounding["annex_iv_checklist"],
            "system_limitations": limitations,
            "limitations_model": limitations,
            "limitations": _ENGINEERING_LIMITATIONS,
            "explicit_limitations": _ENGINEERING_LIMITATIONS,
        }

        pack = _build_model(ExhibitPack, candidates)
        canonical_bytes = canonical_json_bytes(pack)
        digest = sha256_hex(canonical_bytes)
        document = json.loads(canonical_bytes.decode("utf-8"))

        return CompilationResult(
            pack=pack,
            document=document,
            canonical_bytes=canonical_bytes,
            sha256=digest,
            governance_decision=decision,
            standards_matches=flattened_matches,
        )

    def compile_pack(self, *args: Any, **kwargs: Any) -> ExhibitPack:
        return self.compile(*args, **kwargs).pack

    def create_exhibit(self, *args: Any, **kwargs: Any) -> CompilationResult:
        return self.compile(*args, **kwargs)

    def write(
        self,
        result: CompilationResult,
        path: str | os.PathLike[str],
    ) -> CompilationResult:
        if not isinstance(result, CompilationResult):
            raise TypeError("result must be a CompilationResult")
        return result.with_artifact(path)

    def compile_to_path(
        self,
        path: str | os.PathLike[str],
        *args: Any,
        **kwargs: Any,
    ) -> CompilationResult:
        result = self.compile(*args, **kwargs)
        return self.write(result, path)

    def export(
        self,
        path: str | os.PathLike[str],
        *args: Any,
        **kwargs: Any,
    ) -> CompilationResult:
        return self.compile_to_path(path, *args, **kwargs)


def compile_exhibit_pack(
    trace: Any,
    evaluation: Any,
    human_review: Any | None = None,
    job_id: str | None = None,
    public_key: bytes | bytearray | memoryview | str | None = None,
    *,
    review: Any | None = None,
    human_review_signoff: Any | None = None,
    catalog: StandardsCatalog | Mapping[str, Any] | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    clock: Any = None,
    policy: GovernancePolicy | None = None,
    compiler: ExhibitPackCompiler | None = None,
    requested_action: PolicyRequestAction | str = PolicyRequestAction.EXPORT,
) -> CompilationResult:
    """Compile a governed evidence pack using a supplied or default compiler."""

    active_compiler = compiler or ExhibitPackCompiler(
        catalog=catalog,
        catalog_path=catalog_path,
        public_key=public_key,
        clock=clock,
        policy=policy,
    )
    return active_compiler.compile(
        trace,
        evaluation,
        human_review,
        job_id,
        public_key,
        review=review,
        human_review_signoff=human_review_signoff,
        requested_action=requested_action,
        policy=policy,
    )


def _artifact_bytes(payload: Any) -> bytes:
    if isinstance(payload, CompilationResult):
        return payload.canonical_bytes
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return canonical_json_bytes(payload)


def _verify_expected_digest(expected: str | None, actual: str) -> None:
    if expected is None:
        return
    normalised = _normalised_hash(expected)
    if normalised is None:
        raise ImmutableArtifactError("expected artifact SHA-256 is invalid")
    if normalised != actual:
        raise ImmutableArtifactError(
            f"artifact SHA-256 mismatch: expected {normalised}, calculated {actual}"
        )


def write_immutable_artifact(
    path: str | os.PathLike[str],
    payload: Any,
    expected_sha256: str | None = None,
    *,
    artifact_sha256: str | None = None,
    expected_digest: str | None = None,
) -> str:
    """Create a canonical artifact with exclusive creation and durable fsync."""

    destination = Path(path)
    supplied_digests = [
        value
        for value in (expected_sha256, artifact_sha256, expected_digest)
        if value is not None
    ]
    normalised_supplied = [_normalised_hash(value) for value in supplied_digests]
    if any(value is None for value in normalised_supplied):
        raise ImmutableArtifactError("expected artifact SHA-256 is invalid")
    if len(set(normalised_supplied)) > 1:
        raise ImmutableArtifactError("conflicting expected artifact digests")

    data = _artifact_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    _verify_expected_digest(
        normalised_supplied[0] if normalised_supplied else None,
        digest,
    )

    parent = destination.parent
    if not parent.exists():
        raise ArtifactWriteError(f"artifact parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise ArtifactWriteError(f"artifact parent is not a directory: {parent}")

    current = parent
    while True:
        if current.is_symlink():
            raise ArtifactWriteError(f"symbolic-link artifact parent is prohibited: {current}")
        if current.parent == current:
            break
        current = current.parent

    if os.path.lexists(destination):
        raise ArtifactExistsError(f"immutable artifact already exists: {destination}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor: int | None = None
    created = False
    try:
        try:
            descriptor = os.open(destination, flags, 0o444)
            created = True
            os.fchmod(descriptor, 0o444)
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("artifact write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        try:
            directory_descriptor = os.open(parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            if created:
                destination.unlink(missing_ok=True)
            raise ArtifactWriteError("artifact directory could not be durably synchronized")

        written_digest = verify_immutable_artifact(destination, expected_sha256=digest)
        if written_digest != digest:
            raise ImmutableArtifactError("artifact changed while being written")
        return digest
    except (ArtifactExistsError, ImmutableArtifactError):
        if created and descriptor is None:
            try:
                if destination.exists() and not destination.is_symlink():
                    destination.unlink()
            except OSError:
                pass
        raise
    except OSError as error:
        if created:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise ArtifactWriteError(f"immutable artifact could not be written: {destination}") from error


def verify_immutable_artifact(
    path: str | os.PathLike[str],
    expected_sha256: str | None = None,
    *,
    artifact_sha256: str | None = None,
    require_read_only: bool = True,
) -> str:
    """Verify path type, read-only state, JSON syntax, and exact SHA-256."""

    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError as error:
        raise ImmutableArtifactError(f"artifact does not exist: {source}") from error
    except OSError as error:
        raise ImmutableArtifactError(f"artifact metadata cannot be read: {source}") from error

    if stat.S_ISLNK(metadata.st_mode):
        raise ImmutableArtifactError("artifact symbolic links are prohibited")
    if not stat.S_ISREG(metadata.st_mode):
        raise ImmutableArtifactError("artifact must be a regular file")
    if require_read_only and stat.S_IMODE(metadata.st_mode) & 0o222:
        raise ImmutableArtifactError("artifact must not be writable by any class")

    try:
        data = source.read_bytes()
    except OSError as error:
        raise ImmutableArtifactError(f"artifact content cannot be read: {source}") from error

    digest = hashlib.sha256(data).hexdigest()
    _verify_expected_digest(expected_sha256 or artifact_sha256, digest)

    try:
        text = data.decode("utf-8")
        json.loads(
            text,
            object_pairs_hook=lambda pairs: dict(pairs),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ImmutableArtifactError("artifact is not valid UTF-8 JSON") from error

    return digest