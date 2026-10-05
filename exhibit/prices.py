"""Deterministic token extraction and EUR evaluation-cost accounting.

The module treats provider token counts as accounting evidence and never
silently converts provider-native monetary amounts. All rates supplied to
this module must be explicitly denominated in euros per one million tokens.

Provider usage conventions differ:

* OpenAI-style APIs report ``prompt_tokens`` and ``completion_tokens``.
* OpenRouter commonly exposes the same OpenAI-style fields.
* Gemini reports camel-cased fields such as ``promptTokenCount``.
* Anthropic-style APIs report ``input_tokens`` and ``output_tokens``.
* Cached-input and reasoning counts are subsets of their parent categories
  and are never added to ``total_tokens`` a second time.

All calculations use :class:`~decimal.Decimal` and deterministic half-up
rounding to twelve decimal places. Network access is never performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


EUR = "EUR"
TOKENS_PER_PRICE_UNIT = Decimal("1000000")
EUR_ACCOUNTING_QUANTUM = Decimal("0.000000000001")

MAX_EXACT_TOKEN_COUNT = 9_007_199_254_740_991
MAX_RATE_EUR_PER_MILLION = Decimal("1000000000000")
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024

_MISSING = object()

_EUR_DECIMAL_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,18})?(?:[eE][+-]?[0-9]{1,6})?$"
)


class PriceAccountingError(ValueError):
    """Raised when token or monetary evidence is ambiguous or invalid."""


def _normalise_key(value: object) -> str:
    """Return a case-insensitive, punctuation-insensitive mapping key."""

    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _coerce_decimal(
    value: Any,
    *,
    label: str = "amount",
    maximum: Decimal | None = MAX_RATE_EUR_PER_MILLION,
) -> Decimal:
    """Parse a finite, non-negative decimal without using binary floats."""

    if isinstance(value, bool):
        raise PriceAccountingError(f"{label} must be numeric, not boolean")

    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise PriceAccountingError(f"{label} must be finite")
        result = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("€"):
            text = text[1:].strip()
        text = re.sub(r"^EUR\s*", "", text, flags=re.IGNORECASE).strip()
        if not _EUR_DECIMAL_PATTERN.fullmatch(text):
            raise PriceAccountingError(
                f"{label} must be a finite decimal amount, got {value!r}"
            )
        try:
            result = Decimal(text)
        except InvalidOperation as exc:
            raise PriceAccountingError(f"{label} is not a valid decimal") from exc
    else:
        raise PriceAccountingError(
            f"{label} must be a decimal-compatible value, got {type(value).__name__}"
        )

    if not result.is_finite():
        raise PriceAccountingError(f"{label} must be finite")
    if result < 0:
        raise PriceAccountingError(f"{label} cannot be negative")
    if maximum is not None and result.copy_abs() > maximum:
        raise PriceAccountingError(
            f"{label} exceeds the supported maximum of {maximum}"
        )
    return result


def _coerce_int(value: Any, *, label: str = "token count") -> int:
    """Parse an exact non-negative integer token count."""

    if isinstance(value, bool):
        raise PriceAccountingError(f"{label} must be an integer, not boolean")

    if isinstance(value, int):
        result = value
    elif isinstance(value, Decimal):
        if value != value.to_integral_value():
            raise PriceAccountingError(f"{label} must be an integer")
        result = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise PriceAccountingError(f"{label} must be an integer")
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"\+?[0-9]+", text):
            raise PriceAccountingError(
                f"{label} must be a non-negative integer, got {value!r}"
            )
        result = int(text, 10)
    else:
        raise PriceAccountingError(
            f"{label} must be an integer-compatible value, "
            f"got {type(value).__name__}"
        )

    if result < 0:
        raise PriceAccountingError(f"{label} cannot be negative")
    if result > MAX_EXACT_TOKEN_COUNT:
        raise PriceAccountingError(
            f"{label} exceeds the exact integer limit {MAX_EXACT_TOKEN_COUNT}"
        )
    return result


def _round_money(value: Decimal) -> Decimal:
    """Round an amount to the fixed EUR accounting quantum."""

    try:
        with localcontext() as context:
            context.prec = 80
            rounded = value.quantize(EUR_ACCOUNTING_QUANTUM, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise PriceAccountingError("monetary result cannot be represented") from exc

    if rounded == 0:
        return Decimal("0.000000000000")
    return rounded


def _normalised_aliases(aliases: Iterable[str]) -> frozenset[str]:
    return frozenset(_normalise_key(alias) for alias in aliases)


def _read_raw_alias(
    mapping: Mapping[Any, Any], aliases: Iterable[str]
) -> Any | None:
    """Return the first present value for a normalized alias set."""

    normalized = _normalised_aliases(aliases)
    for key, value in mapping.items():
        if value is not None and _normalise_key(key) in normalized:
            return value
    return None


def _read_alias(
    mapping: Mapping[Any, Any],
    aliases: Iterable[str],
    parser: Callable[[Any], Any],
    *,
    label: str,
) -> Any | None:
    """Read aliases and reject conflicting duplicate values."""

    normalized = _normalised_aliases(aliases)
    parsed_values: list[Any] = []
    for key, value in mapping.items():
        if value is None or _normalise_key(key) not in normalized:
            continue
        parsed_values.append(parser(value))

    if not parsed_values:
        return None

    first = parsed_values[0]
    if any(value != first for value in parsed_values[1:]):
        keys = ", ".join(str(alias) for alias in aliases)
        raise PriceAccountingError(
            f"conflicting values were supplied for {label} aliases ({keys})"
        )
    return first


def _read_text_alias(
    mapping: Mapping[Any, Any], aliases: Iterable[str], *, label: str
) -> str | None:
    value = _read_alias(
        mapping,
        aliases,
        lambda item: str(item).strip(),
        label=label,
    )
    if value is not None and not value:
        raise PriceAccountingError(f"{label} cannot be empty")
    return value


_INPUT_ALIASES = (
    "input_tokens",
    "prompt_tokens",
    "input_token_count",
    "prompt_token_count",
    "inputTokenCount",
    "promptTokenCount",
    "inputTokens",
    "promptTokens",
)
_OUTPUT_ALIASES = (
    "output_tokens",
    "completion_tokens",
    "output_token_count",
    "completion_token_count",
    "outputTokenCount",
    "completionTokenCount",
    "candidatesTokenCount",
    "outputTokens",
    "completionTokens",
)
_TOTAL_ALIASES = (
    "total_tokens",
    "total_token_count",
    "totalTokenCount",
    "totalTokens",
)
_CACHED_ALIASES = (
    "cached_tokens",
    "cached_input_tokens",
    "cached_content_token_count",
    "cachedContentTokenCount",
    "cache_read_input_tokens",
    "cache_read_tokens",
    "prompt_cache_hit_tokens",
    "cacheReadInputTokens",
    "cacheReadTokens",
)
_CACHE_CREATION_ALIASES = (
    "cache_creation_input_tokens",
    "cache_creation_tokens",
    "cacheCreationInputTokens",
)
_REASONING_ALIASES = (
    "reasoning_tokens",
    "reasoning_token_count",
    "reasoningTokenCount",
    "thoughts_token_count",
    "thoughtsTokenCount",
    "thinking_tokens",
    "thoughtsTokens",
)
_DETAIL_ALIASES = (
    "prompt_tokens_details",
    "input_tokens_details",
    "completion_tokens_details",
    "output_tokens_details",
    "token_details",
)

_DIRECT_TOKEN_KEYS = frozenset(
    _normalised_aliases(
        _INPUT_ALIASES
        + _OUTPUT_ALIASES
        + _TOTAL_ALIASES
        + _CACHED_ALIASES
        + _CACHE_CREATION_ALIASES
        + _REASONING_ALIASES
    )
)
_DETAIL_KEYS = _normalised_aliases(_DETAIL_ALIASES)
_ALL_TOKEN_KEYS = _DIRECT_TOKEN_KEYS | _DETAIL_KEYS

_USAGE_CONTAINER_ALIASES = (
    "usage",
    "usage_metadata",
    "usageMetadata",
    "token_usage",
    "tokenUsage",
    "token_usage_metadata",
    "response_usage",
    "responseUsage",
)
_USAGE_CONTAINER_KEYS = _normalised_aliases(_USAGE_CONTAINER_ALIASES)

_TOKEN_METADATA_KEYS = _normalised_aliases(
    (
        "id",
        "model",
        "object",
        "provider",
        "finish_reason",
        "finishReason",
        "system_fingerprint",
        "systemFingerprint",
        "created",
        "response_id",
        "responseId",
        "cost",
        "cost_details",
        "costDetails",
        "input_cost",
        "output_cost",
        "total_cost",
        "input_cost_eur",
        "output_cost_eur",
        "total_cost_eur",
        "currency",
    )
)


def _token_value(
    data: Mapping[Any, Any],
    aliases: Iterable[str],
    *,
    label: str,
    include_details: bool = True,
) -> int | None:
    """Read a token count from a usage object and its detail objects."""

    direct = _read_alias(
        data,
        aliases,
        lambda item: _coerce_int(item, label=label),
        label=label,
    )
    if direct is not None or not include_details:
        return direct

    detail_values: list[int] = []
    for detail_alias in _DETAIL_ALIASES:
        details = _read_raw_alias(data, (detail_alias,))
        if isinstance(details, Mapping):
            nested = _read_alias(
                details,
                aliases,
                lambda item: _coerce_int(item, label=label),
                label=label,
            )
            if nested is not None:
                detail_values.append(nested)

    if not detail_values:
        return None
    first = detail_values[0]
    if any(value != first for value in detail_values[1:]):
        raise PriceAccountingError(
            f"conflicting {label} values were supplied in detail objects"
        )
    return first


def _cached_token_value(data: Mapping[Any, Any]) -> int:
    direct = _token_value(
        data,
        _CACHED_ALIASES,
        label="cached input token count",
    )
    creation = _read_alias(
        data,
        _CACHE_CREATION_ALIASES,
        lambda item: _coerce_int(item, label="cache creation token count"),
        label="cache creation token count",
    )

    if direct is None:
        return creation or 0
    if creation is None:
        return direct
    return direct + creation


def _looks_like_usage(data: Mapping[Any, Any]) -> bool:
    keys = {_normalise_key(key) for key in data}
    return bool(keys & _ALL_TOKEN_KEYS)


def _decode_json_value(value: Any) -> Any | None:
    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) > MAX_PAYLOAD_BYTES:
                return None
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    if len(stripped.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _payload_to_mapping(payload: Any) -> Mapping[Any, Any] | list[Any]:
    """Convert supported response containers into a bounded JSON-like tree."""

    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise PriceAccountingError("token-usage payload exceeds the size limit")
        try:
            payload = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PriceAccountingError("token-usage payload is not UTF-8") from exc

    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_PAYLOAD_BYTES:
            raise PriceAccountingError("token-usage payload exceeds the size limit")
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PriceAccountingError(
                "token-usage payload must be valid JSON"
            ) from exc

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="python")

    if is_dataclass(payload):
        payload = asdict(payload)

    if isinstance(payload, Mapping):
        return payload

    if isinstance(payload, list):
        return payload

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(payload, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, Mapping):
                return converted

    json_method = getattr(payload, "json", None)
    if callable(json_method):
        converted = json_method()
        if isinstance(converted, Mapping):
            return converted

    if hasattr(payload, "usage"):
        return {"usage": getattr(payload, "usage")}

    raise PriceAccountingError(
        "token usage must be supplied as JSON, bytes, a mapping, "
        "or a response object"
    )


def _locate_usage(
    payload: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Mapping[Any, Any] | None:
    """Find a provider usage object without trusting a monetary cost field."""

    if _depth > 20:
        return None
    if _seen is None:
        _seen = set()

    if isinstance(payload, (bytes, bytearray, memoryview, str)):
        payload = _decode_json_value(payload)
        if payload is None:
            return None

    if isinstance(payload, Mapping):
        identity = id(payload)
        if identity in _seen:
            return None
        _seen.add(identity)

        for key, value in payload.items():
            if _normalise_key(key) not in _USAGE_CONTAINER_KEYS:
                continue
            if isinstance(value, Mapping):
                nested = _locate_usage(
                    value, _depth=_depth + 1, _seen=_seen
                )
                if nested is not None:
                    return nested
                if _looks_like_usage(value):
                    return value
            else:
                nested = _locate_usage(
                    value, _depth=_depth + 1, _seen=_seen
                )
                if nested is not None:
                    return nested

        if _looks_like_usage(payload):
            return payload

        for value in payload.values():
            nested = _locate_usage(value, _depth=_depth + 1, _seen=_seen)
            if nested is not None:
                return nested
        return None

    if isinstance(payload, list):
        for value in payload:
            nested = _locate_usage(value, _depth=_depth + 1, _seen=_seen)
            if nested is not None:
                return nested

    return None


def _normalise_token_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise ValueError("token usage must be a mapping")

    located = _locate_usage(value)
    source = value if located is None else located
    data = dict(source)

    try:
        input_tokens = _token_value(
            data, _INPUT_ALIASES, label="input token count"
        )
        output_tokens = _token_value(
            data, _OUTPUT_ALIASES, label="output token count"
        )
        total_tokens = _token_value(
            data,
            _TOTAL_ALIASES,
            label="total token count",
            include_details=False,
        )
        cached_tokens = _cached_token_value(data)
        reasoning_tokens = _token_value(
            data,
            _REASONING_ALIASES,
            label="reasoning token count",
        )
    except PriceAccountingError as exc:
        raise ValueError(str(exc)) from exc

    if input_tokens is None and output_tokens is None and total_tokens is not None:
        input_tokens = total_tokens
    if input_tokens is None:
        input_tokens = 0
    if output_tokens is None:
        output_tokens = 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    elif total_tokens != input_tokens + output_tokens:
        raise ValueError(
            "total_tokens must equal input_tokens + output_tokens; "
            "cached and reasoning counts are subsets"
        )

    for key in list(data):
        normalized = _normalise_key(key)
        if normalized in _ALL_TOKEN_KEYS or normalized in _TOKEN_METADATA_KEYS:
            data.pop(key, None)
        elif normalized in _USAGE_CONTAINER_KEYS:
            data.pop(key, None)

    data["input_tokens"] = input_tokens
    data["output_tokens"] = output_tokens
    data["total_tokens"] = total_tokens
    data["cached_input_tokens"] = cached_tokens
    data["reasoning_tokens"] = reasoning_tokens or 0
    return data


class TokenUsage(BaseModel):
    """Canonical, immutable token counts for one or more model calls.

    ``cached_input_tokens`` is a subset of ``input_tokens`` and
    ``reasoning_tokens`` is a subset of ``output_tokens``. Therefore,
    ``total_tokens`` is always ``input_tokens + output_tokens``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
        validate_default=True,
    )

    input_tokens: int = Field(default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, strict=True)
    output_tokens: int = Field(default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, strict=True)
    total_tokens: int = Field(default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, strict=True)
    cached_input_tokens: int = Field(
        default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, strict=True
    )
    reasoning_tokens: int = Field(
        default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, strict=True
    )

    @model_validator(mode="before")
    @classmethod
    def normalise_provider_fields(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        return _normalise_token_mapping(value)

    @model_validator(mode="after")
    def validate_relationships(self) -> TokenUsage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError(
                "total_tokens must equal input_tokens + output_tokens"
            )
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")
        return self

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @property
    def total(self) -> int:
        return self.total_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cached_input_tokens

    @property
    def non_reasoning_output_tokens(self) -> int:
        return self.output_tokens - self.reasoning_tokens

    @property
    def billable_input_tokens(self) -> int:
        return self.uncached_input_tokens

    @property
    def billable_output_tokens(self) -> int:
        return self.non_reasoning_output_tokens

    def to_dict(self) -> dict[str, int]:
        return self.model_dump(mode="json")

    def as_dict(self) -> dict[str, int]:
        return self.to_dict()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TokenUsage:
        return cls.model_validate(value)

    @classmethod
    def from_response(cls, value: Any) -> TokenUsage:
        return extract_token_usage(value)

    def __str__(self) -> str:
        return (
            f"{self.input_tokens} input + {self.output_tokens} output "
            f"= {self.total_tokens} total tokens"
        )


def extract_token_usage(
    payload: Any,
    *,
    allow_missing: bool = False,
    allow_total_only: bool = False,
    default: Any = _MISSING,
) -> TokenUsage:
    """Extract canonical token usage from a provider response.

    The function searches common response envelopes and never interprets
    provider monetary fields. If a total is reported, input/output counts
    may be derived from it. A total-only response is rejected by default
    because allocating the total to input or output would invent evidence.
    """

    if default is not _MISSING:
        if default is None:
            return TokenUsage()
        return _coerce_usage(default)

    if isinstance(payload, TokenUsage):
        return payload

    root = _payload_to_mapping(payload)
    usage = _locate_usage(root)
    if usage is None:
        if allow_missing:
            return TokenUsage()
        raise PriceAccountingError("no token usage object was found in the payload")

    input_tokens = _token_value(
        usage, _INPUT_ALIASES, label="input token count"
    )
    output_tokens = _token_value(
        usage, _OUTPUT_ALIASES, label="output token count"
    )
    total_tokens = _token_value(
        usage,
        _TOTAL_ALIASES,
        label="total token count",
        include_details=False,
    )
    cached_tokens = _cached_token_value(usage)
    reasoning_tokens = _token_value(
        usage,
        _REASONING_ALIASES,
        label="reasoning token count",
    )

    if input_tokens is None and output_tokens is None:
        if total_tokens is None:
            raise PriceAccountingError(
                "token usage must contain input or output token counts"
            )
        if not allow_total_only:
            raise PriceAccountingError(
                "only total_tokens was reported; input/output allocation "
                "would be ambiguous"
            )
        input_tokens = total_tokens

    if input_tokens is None:
        input_tokens = 0
    if output_tokens is None:
        output_tokens = 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    elif total_tokens != input_tokens + output_tokens:
        raise PriceAccountingError(
            "reported total_tokens does not equal input_tokens + output_tokens"
        )

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens or 0,
    )


def _read_rate_group(
    data: Mapping[Any, Any],
    *,
    million_aliases: Iterable[str],
    thousand_aliases: Iterable[str] = (),
    token_aliases: Iterable[str] = (),
    label: str,
) -> Decimal | None:
    values: list[Decimal] = []

    million = _read_alias(
        data,
        million_aliases,
        lambda item: _coerce_decimal(item, label=label),
        label=label,
    )
    if million is not None:
        values.append(million)

    thousand = _read_alias(
        data,
        thousand_aliases,
        lambda item: _coerce_decimal(item, label=f"{label} per 1K"),
        label=f"{label} per 1K",
    )
    if thousand is not None:
        converted = thousand * Decimal("1000")
        if converted > MAX_RATE_EUR_PER_MILLION:
            raise PriceAccountingError(
                f"{label} per-1K rate converts above the EUR rate limit"
            )
        values.append(converted)

    per_token = _read_alias(
        data,
        token_aliases,
        lambda item: _coerce_decimal(item, label=f"{label} per token"),
        label=f"{label} per token",
    )
    if per_token is not None:
        converted = per_token * TOKENS_PER_PRICE_UNIT
        if converted > MAX_RATE_EUR_PER_MILLION:
            raise PriceAccountingError(
                f"{label} per-token rate converts above the EUR rate limit"
            )
        values.append(converted)

    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise PriceAccountingError(
            f"conflicting unit variants were supplied for {label}"
        )
    return first


_INPUT_MILLION_ALIASES = (
    "input_eur_per_million_tokens",
    "input_cost_per_million_eur",
    "input_cost_per_1m_eur",
    "input_price_eur_per_million_tokens",
    "input_rate_eur_per_million_tokens",
    "input_eur_per_million",
    "input_per_million_eur",
    "input_per_million_tokens_eur",
    "prompt_eur_per_million_tokens",
    "prompt_cost_per_million_eur",
    "input_rate",
    "input_cost_per_million",
)
_OUTPUT_MILLION_ALIASES = (
    "output_eur_per_million_tokens",
    "output_cost_per_million_eur",
    "output_cost_per_1m_eur",
    "output_price_eur_per_million_tokens",
    "output_rate_eur_per_million_tokens",
    "output_eur_per_million",
    "output_per_million_eur",
    "output_per_million_tokens_eur",
    "completion_eur_per_million_tokens",
    "completion_cost_per_million_eur",
    "output_rate",
    "output_cost_per_million",
)
_CACHED_MILLION_ALIASES = (
    "cached_input_eur_per_million_tokens",
    "cached_input_cost_per_million_eur",
    "cached_input_rate_eur_per_million_tokens",
    "cached_input_eur_per_million",
    "cached_rate_eur_per_million_tokens",
    "cached_input_rate",
)
_REASONING_MILLION_ALIASES = (
    "reasoning_eur_per_million_tokens",
    "reasoning_output_eur_per_million_tokens",
    "reasoning_cost_per_million_eur",
    "reasoning_rate_eur_per_million_tokens",
    "reasoning_eur_per_million",
    "reasoning_rate_eur_per_million",
    "reasoning_rate",
)
_INPUT_THOUSAND_ALIASES = (
    "input_eur_per_1k_tokens",
    "input_cost_per_1k_eur",
    "input_price_eur_per_1k_tokens",
    "input_per_1k_eur",
)
_OUTPUT_THOUSAND_ALIASES = (
    "output_eur_per_1k_tokens",
    "output_cost_per_1k_eur",
    "output_price_eur_per_1k_tokens",
    "output_per_1k_eur",
)
_CACHED_THOUSAND_ALIASES = (
    "cached_input_eur_per_1k_tokens",
    "cached_input_cost_per_1k_eur",
)
_REASONING_THOUSAND_ALIASES = (
    "reasoning_eur_per_1k_tokens",
    "reasoning_cost_per_1k_eur",
)
_INPUT_TOKEN_ALIASES = (
    "input_eur_per_token",
    "input_cost_per_token_eur",
    "input_price_eur_per_token",
)
_OUTPUT_TOKEN_ALIASES = (
    "output_eur_per_token",
    "output_cost_per_token_eur",
    "output_price_eur_per_token",
)
_CACHED_TOKEN_ALIASES = (
    "cached_input_eur_per_token",
    "cached_input_cost_per_token_eur",
)
_REASONING_TOKEN_ALIASES = (
    "reasoning_eur_per_token",
    "reasoning_cost_per_token_eur",
)
_MODEL_ALIASES = ("model", "model_id", "model_name", "modelId", "modelName")
_CURRENCY_ALIASES = ("currency", "currency_code", "price_currency", "currencyCode")
_PRICING_CONTAINER_ALIASES = (
    "pricing",
    "price_card",
    "priceCard",
    "rate_card",
    "rateCard",
)
_PRICING_RECOGNISED_KEYS = _normalised_aliases(
    _INPUT_MILLION_ALIASES
    + _OUTPUT_MILLION_ALIASES
    + _CACHED_MILLION_ALIASES
    + _REASONING_MILLION_ALIASES
    + _INPUT_THOUSAND_ALIASES
    + _OUTPUT_THOUSAND_ALIASES
    + _CACHED_THOUSAND_ALIASES
    + _REASONING_THOUSAND_ALIASES
    + _INPUT_TOKEN_ALIASES
    + _OUTPUT_TOKEN_ALIASES
    + _CACHED_TOKEN_ALIASES
    + _REASONING_TOKEN_ALIASES
    + _MODEL_ALIASES
    + _CURRENCY_ALIASES
    + _PRICING_CONTAINER_ALIASES
    + ("id", "provider", "source", "created", "effective_at")
)


def _normalise_pricing_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise ValueError("model pricing must be supplied as a mapping")

    data = dict(value)
    nested = _read_raw_alias(data, _PRICING_CONTAINER_ALIASES)
    if isinstance(nested, Mapping):
        merged = dict(nested)
        for key, item in data.items():
            if _normalise_key(key) not in _normalised_aliases(
                _PRICING_CONTAINER_ALIASES
            ):
                merged[key] = item
        data = merged

    try:
        model = _read_text_alias(data, _MODEL_ALIASES, label="model identifier")
        currency = _read_text_alias(
            data, _CURRENCY_ALIASES, label="currency"
        )
        input_rate = _read_rate_group(
            data,
            million_aliases=_INPUT_MILLION_ALIASES,
            thousand_aliases=_INPUT_THOUSAND_ALIASES,
            token_aliases=_INPUT_TOKEN_ALIASES,
            label="input EUR rate",
        )
        output_rate = _read_rate_group(
            data,
            million_aliases=_OUTPUT_MILLION_ALIASES,
            thousand_aliases=_OUTPUT_THOUSAND_ALIASES,
            token_aliases=_OUTPUT_TOKEN_ALIASES,
            label="output EUR rate",
        )
        cached_rate = _read_rate_group(
            data,
            million_aliases=_CACHED_MILLION_ALIASES,
            thousand_aliases=_CACHED_THOUSAND_ALIASES,
            token_aliases=_CACHED_TOKEN_ALIASES,
            label="cached-input EUR rate",
        )
        reasoning_rate = _read_rate_group(
            data,
            million_aliases=_REASONING_MILLION_ALIASES,
            thousand_aliases=_REASONING_THOUSAND_ALIASES,
            token_aliases=_REASONING_TOKEN_ALIASES,
            label="reasoning-output EUR rate",
        )
    except PriceAccountingError as exc:
        raise ValueError(str(exc)) from exc

    if currency is None:
        currency = EUR
    currency = currency.upper()
    if currency in {"€", "EURO"}:
        currency = EUR
    if currency != EUR:
        raise ValueError(
            f"only {EUR} pricing is supported; provider-native costs "
            f"in {currency} are not accepted"
        )

    for key in list(data):
        if _normalise_key(key) in _PRICING_RECOGNISED_KEYS:
            data.pop(key, None)

    data["model"] = model or "unspecified"
    data["currency"] = currency
    if input_rate is not None:
        data["input_eur_per_million_tokens"] = input_rate
    if output_rate is not None:
        data["output_eur_per_million_tokens"] = output_rate
    if cached_rate is not None:
        data["cached_input_eur_per_million_tokens"] = cached_rate
    if reasoning_rate is not None:
        data["reasoning_eur_per_million_tokens"] = reasoning_rate
    return data


class ModelPricing(BaseModel):
    """Reviewed model rates denominated in EUR per one million tokens."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=False,
        validate_default=True,
    )

    model: str = Field(default="unspecified", min_length=1, max_length=256)
    input_eur_per_million_tokens: Decimal = Field(
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
    )
    output_eur_per_million_tokens: Decimal = Field(
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
    )
    cached_input_eur_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
    )
    reasoning_eur_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
    )
    currency: Literal["EUR"] = EUR

    @model_validator(mode="before")
    @classmethod
    def normalise_pricing_fields(cls, value: Any) -> Any:
        return _normalise_pricing_mapping(value)

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, value: Any) -> str:
        if value is None:
            return "unspecified"
        text = str(value).strip()
        if not text:
            raise ValueError("model identifier cannot be empty")
        return text

    @field_validator(
        "input_eur_per_million_tokens",
        "output_eur_per_million_tokens",
        "cached_input_eur_per_million_tokens",
        "reasoning_eur_per_million_tokens",
        mode="before",
    )
    @classmethod
    def validate_rate(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return _coerce_decimal(value, label="EUR token rate")

    @field_validator("currency", mode="before")
    @classmethod
    def normalise_currency(cls, value: Any) -> str:
        if value is None:
            return EUR
        text = str(value).strip().upper()
        return EUR if text in {"€", "EURO"} else text

    @property
    def input_cost_per_million_eur(self) -> Decimal:
        return self.input_eur_per_million_tokens

    @property
    def output_cost_per_million_eur(self) -> Decimal:
        return self.output_eur_per_million_tokens

    @property
    def input_rate_eur_per_million(self) -> Decimal:
        return self.input_eur_per_million_tokens

    @property
    def output_rate_eur_per_million(self) -> Decimal:
        return self.output_eur_per_million_tokens

    @property
    def input_eur_per_million(self) -> Decimal:
        return self.input_eur_per_million_tokens

    @property
    def output_eur_per_million(self) -> Decimal:
        return self.output_eur_per_million_tokens

    @property
    def cached_input_rate(self) -> Decimal:
        if self.cached_input_eur_per_million_tokens is None:
            return self.input_eur_per_million_tokens
        return self.cached_input_eur_per_million_tokens

    @property
    def reasoning_rate(self) -> Decimal:
        if self.reasoning_eur_per_million_tokens is None:
            return self.output_eur_per_million_tokens
        return self.reasoning_eur_per_million_tokens

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


_COST_INPUT_ALIASES = (
    "input_cost_eur",
    "input_cost",
    "input_eur",
    "input_price_eur",
)
_COST_OUTPUT_ALIASES = (
    "output_cost_eur",
    "output_cost",
    "output_eur",
    "output_price_eur",
)
_COST_TOTAL_ALIASES = (
    "total_cost_eur",
    "total_cost",
    "total_eur",
    "total_price_eur",
)
_COST_CACHED_ALIASES = (
    "cached_input_cost_eur",
    "cached_input_cost",
    "cached_input_eur",
)
_COST_REASONING_ALIASES = (
    "reasoning_output_cost_eur",
    "reasoning_output_cost",
    "reasoning_eur",
)
_COST_UNCACHED_ALIASES = (
    "uncached_input_cost_eur",
    "uncached_input_cost",
    "uncached_input_eur",
)
_COST_NON_REASONING_ALIASES = (
    "non_reasoning_output_cost_eur",
    "non_reasoning_output_cost",
    "non_reasoning_output_eur",
)
_COST_RATE_ALIASES = (
    "input_rate_eur_per_million_tokens",
    "output_rate_eur_per_million_tokens",
    "cached_input_rate_eur_per_million_tokens",
    "reasoning_rate_eur_per_million_tokens",
)
_COST_COUNT_ALIASES = (
    "evaluation_count",
    "evaluations",
    "count",
    "n_evaluations",
    "number_of_evaluations",
)
_COST_RECOGNISED_KEYS = _normalised_aliases(
    _COST_INPUT_ALIASES
    + _COST_OUTPUT_ALIASES
    + _COST_TOTAL_ALIASES
    + _COST_CACHED_ALIASES
    + _COST_REASONING_ALIASES
    + _COST_UNCACHED_ALIASES
    + _COST_NON_REASONING_ALIASES
    + _COST_RATE_ALIASES
    + _COST_COUNT_ALIASES
    + _MODEL_ALIASES
    + _CURRENCY_ALIASES
    + _USAGE_CONTAINER_ALIASES
    + ("token_usage", "usage", "id", "provider", "source")
)


def _cost_amount(
    data: Mapping[Any, Any], aliases: Iterable[str], *, label: str
) -> Decimal | None:
    value = _read_alias(
        data,
        aliases,
        lambda item: _coerce_decimal(item, label=label, maximum=None),
        label=label,
    )
    return value


def _normalise_cost_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise ValueError("cost breakdown must be supplied as a mapping")

    data = dict(value)
    raw_usage = _read_raw_alias(
        data,
        _USAGE_CONTAINER_ALIASES + ("token_usage",),
    )
    usage_value = _coerce_usage(raw_usage) if raw_usage is not None else None

    input_tokens = _token_value(data, _INPUT_ALIASES, label="input token count")
    output_tokens = _token_value(data, _OUTPUT_ALIASES, label="output token count")
    total_tokens = _token_value(
        data,
        _TOTAL_ALIASES,
        label="total token count",
        include_details=False,
    )

    if input_tokens is None and usage_value is not None:
        input_tokens = usage_value.input_tokens
    if output_tokens is None and usage_value is not None:
        output_tokens = usage_value.output_tokens
    if total_tokens is None and usage_value is not None:
        total_tokens = usage_value.total_tokens
    if input_tokens is None:
        input_tokens = 0
    if output_tokens is None:
        output_tokens = 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    elif total_tokens != input_tokens + output_tokens:
        raise ValueError(
            "cost breakdown total_tokens must equal input_tokens + output_tokens"
        )

    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=(
            usage_value.cached_input_tokens if usage_value is not None else 0
        ),
        reasoning_tokens=(
            usage_value.reasoning_tokens if usage_value is not None else 0
        ),
    )

    input_cost = _cost_amount(
        data, _COST_INPUT_ALIASES, label="input cost"
    )
    output_cost = _cost_amount(
        data, _COST_OUTPUT_ALIASES, label="output cost"
    )
    total_cost = _cost_amount(
        data, _COST_TOTAL_ALIASES, label="total cost"
    )
    uncached_input_cost = _cost_amount(
        data, _COST_UNCACHED_ALIASES, label="uncached input cost"
    )
    cached_input_cost = _cost_amount(
        data, _COST_CACHED_ALIASES, label="cached input cost"
    )
    non_reasoning_output_cost = _cost_amount(
        data,
        _COST_NON_REASONING_ALIASES,
        label="non-reasoning output cost",
    )
    reasoning_output_cost = _cost_amount(
        data, _COST_REASONING_ALIASES, label="reasoning output cost"
    )

    if input_cost is None:
        if uncached_input_cost is not None or cached_input_cost is not None:
            input_cost = (uncached_input_cost or Decimal("0")) + (
                cached_input_cost or Decimal("0")
            )
        elif total_cost is not None and output_cost is not None:
            input_cost = total_cost - output_cost
        else:
            input_cost = Decimal("0")

    if output_cost is None:
        if non_reasoning_output_cost is not None or reasoning_output_cost is not None:
            output_cost = (non_reasoning_output_cost or Decimal("0")) + (
                reasoning_output_cost or Decimal("0")
            )
        elif total_cost is not None and input_cost is not None:
            output_cost = total_cost - input_cost
        else:
            output_cost = Decimal("0")

    if input_cost < 0 or output_cost < 0:
        raise ValueError("cost components cannot be negative")
    if total_cost is None:
        total_cost = input_cost + output_cost
    elif total_cost != input_cost + output_cost:
        raise ValueError("total_cost_eur must equal input and output cost components")

    model = _read_text_alias(data, _MODEL_ALIASES, label="model identifier")
    currency = _read_text_alias(data, _CURRENCY_ALIASES, label="currency")
    currency = (currency or EUR).upper()
    if currency in {"€", "EURO"}:
        currency = EUR
    if currency != EUR:
        raise ValueError("cost breakdown currency must be EUR")

    input_rate = _cost_amount(
        data,
        ("input_rate_eur_per_million_tokens",),
        label="input EUR rate",
    )
    output_rate = _cost_amount(
        data,
        ("output_rate_eur_per_million_tokens",),
        label="output EUR rate",
    )
    cached_rate = _cost_amount(
        data,
        ("cached_input_rate_eur_per_million_tokens",),
        label="cached-input EUR rate",
    )
    reasoning_rate = _cost_amount(
        data,
        ("reasoning_rate_eur_per_million_tokens",),
        label="reasoning EUR rate",
    )

    count = _read_alias(
        data,
        _COST_COUNT_ALIASES,
        lambda item: _coerce_int(item, label="evaluation count"),
        label="evaluation count",
    )
    if count is None:
        count = 1

    for key in list(data):
        if _normalise_key(key) in _COST_RECOGNISED_KEYS:
            data.pop(key, None)

    data["input_tokens"] = usage.input_tokens
    data["output_tokens"] = usage.output_tokens
    data["total_tokens"] = usage.total_tokens
    data["cached_input_tokens"] = usage.cached_input_tokens
    data["reasoning_tokens"] = usage.reasoning_tokens
    data["input_cost_eur"] = _round_money(input_cost)
    data["output_cost_eur"] = _round_money(output_cost)
    data["total_cost_eur"] = _round_money(total_cost)
    data["uncached_input_cost_eur"] = _round_money(
        uncached_input_cost or Decimal("0")
    )
    data["cached_input_cost_eur"] = _round_money(
        cached_input_cost or Decimal("0")
    )
    data["non_reasoning_output_cost_eur"] = _round_money(
        non_reasoning_output_cost or Decimal("0")
    )
    data["reasoning_output_cost_eur"] = _round_money(
        reasoning_output_cost or Decimal("0")
    )
    data["evaluation_count"] = count
    data["currency"] = currency
    if model is not None:
        data["model"] = model
    if input_rate is not None:
        data["input_rate_eur_per_million_tokens"] = input_rate
    if output_rate is not None:
        data["output_rate_eur_per_million_tokens"] = output_rate
    if cached_rate is not None:
        data["cached_input_rate_eur_per_million_tokens"] = cached_rate
    if reasoning_rate is not None:
        data["reasoning_rate_eur_per_million_tokens"] = reasoning_rate
    return data


class CostBreakdown(BaseModel):
    """Immutable cost result with explicit EUR denomination."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=False,
        validate_default=True,
    )

    input_tokens: int = Field(default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT)
    output_tokens: int = Field(default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT)
    total_tokens: int = Field(default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT)
    cached_input_tokens: int = Field(
        default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, compare=False
    )
    reasoning_tokens: int = Field(
        default=0, ge=0, le=MAX_EXACT_TOKEN_COUNT, compare=False
    )
    input_cost_eur: Decimal = Field(default=Decimal("0"), ge=0)
    output_cost_eur: Decimal = Field(default=Decimal("0"), ge=0)
    total_cost_eur: Decimal = Field(default=Decimal("0"), ge=0)
    uncached_input_cost_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        compare=False,
        repr=False,
        exclude=True,
    )
    cached_input_cost_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        compare=False,
        repr=False,
        exclude=True,
    )
    non_reasoning_output_cost_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        compare=False,
        repr=False,
        exclude=True,
    )
    reasoning_output_cost_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        compare=False,
        repr=False,
        exclude=True,
    )
    input_rate_eur_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
        compare=False,
        repr=False,
        exclude=True,
    )
    output_rate_eur_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
        compare=False,
        repr=False,
        exclude=True,
    )
    cached_input_rate_eur_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
        compare=False,
        repr=False,
        exclude=True,
    )
    reasoning_rate_eur_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
        le=MAX_RATE_EUR_PER_MILLION,
        compare=False,
        repr=False,
        exclude=True,
    )
    evaluation_count: int = Field(default=1, ge=0, le=MAX_EXACT_TOKEN_COUNT)
    currency: Literal["EUR"] = EUR
    model: str | None = Field(default=None, max_length=256)

    @model_validator(mode="before")
    @classmethod
    def normalise_cost_fields(cls, value: Any) -> Any:
        return _normalise_cost_mapping(value)

    @model_validator(mode="after")
    def validate_cost_relationships(self) -> CostBreakdown:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError(
                "total_tokens must equal input_tokens + output_tokens"
            )
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")
        if self.total_cost_eur != self.input_cost_eur + self.output_cost_eur:
            raise ValueError(
                "total_cost_eur must equal input_cost_eur + output_cost_eur"
            )
        return self

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cached_input_tokens=self.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )

    @property
    def input_cost(self) -> Decimal:
        return self.input_cost_eur

    @property
    def output_cost(self) -> Decimal:
        return self.output_cost_eur

    @property
    def total_cost(self) -> Decimal:
        return self.total_cost_eur

    @property
    def cost_eur(self) -> Decimal:
        return self.total_cost_eur

    @property
    def input_eur(self) -> Decimal:
        return self.input_cost_eur

    @property
    def output_eur(self) -> Decimal:
        return self.output_cost_eur

    @property
    def total_eur(self) -> Decimal:
        return self.total_cost_eur

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def __str__(self) -> str:
        return format_eur(self.total_cost_eur)


def _coerce_usage(value: Any) -> TokenUsage:
    if isinstance(value, TokenUsage):
        return value
    if value is None:
        raise PriceAccountingError("token usage is required")
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        if hasattr(value, "input_tokens") and hasattr(value, "output_tokens"):
            return TokenUsage(
                input_tokens=getattr(value, "input_tokens"),
                output_tokens=getattr(value, "output_tokens"),
                total_tokens=getattr(value, "total_tokens", 0),
                cached_input_tokens=getattr(value, "cached_input_tokens", 0),
                reasoning_tokens=getattr(value, "reasoning_tokens", 0),
            )
        raise PriceAccountingError("token usage has an unsupported type")

    try:
        if _looks_like_usage(value):
            return TokenUsage.model_validate(value)
        return extract_token_usage(value)
    except (PriceAccountingError, ValidationError, ValueError) as exc:
        if isinstance(exc, PriceAccountingError):
            raise
        raise PriceAccountingError(f"invalid token usage: {exc}") from exc


def _coerce_pricing(value: Any) -> ModelPricing:
    if isinstance(value, ModelPricing):
        return value
    if value is None:
        raise PriceAccountingError("EUR model pricing is required")
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise PriceAccountingError("model pricing must be a mapping")

    try:
        return ModelPricing.model_validate(value)
    except (PriceAccountingError, ValidationError, ValueError) as exc:
        if isinstance(exc, PriceAccountingError):
            raise
        raise PriceAccountingError(f"invalid EUR model pricing: {exc}") from exc


def calculate_cost(
    usage: TokenUsage | Mapping[str, Any] | None = None,
    pricing: ModelPricing | Mapping[str, Any] | None = None,
    *,
    token_usage: TokenUsage | Mapping[str, Any] | None = None,
    model_pricing: ModelPricing | Mapping[str, Any] | None = None,
    cached_input_eur_per_million_tokens: Decimal | Mapping[str, Any] | None = None,
    reasoning_eur_per_million_tokens: Decimal | Mapping[str, Any] | None = None,
    cached_input_rate_eur_per_million: Decimal | None = None,
    reasoning_rate_eur_per_million: Decimal | None = None,
) -> CostBreakdown:
    """Calculate one evaluation's cost in EUR.

    Cached input tokens and reasoning output tokens use discounted rates
    when those rates are supplied. Otherwise they use the ordinary input
    and output rates respectively.
    """

    if usage is not None and token_usage is not None:
        raise PriceAccountingError("supply usage or token_usage, not both")
    if pricing is not None and model_pricing is not None:
        raise PriceAccountingError(
            "supply pricing or model_pricing, not both"
        )
    usage = usage if usage is not None else token_usage
    pricing = pricing if pricing is not None else model_pricing
    if usage is None or pricing is None:
        raise PriceAccountingError("both token usage and EUR pricing are required")

    resolved_usage = _coerce_usage(usage)
    resolved_pricing = _coerce_pricing(pricing)

    input_rate = resolved_pricing.input_eur_per_million_tokens
    output_rate = resolved_pricing.output_eur_per_million_tokens
    cached_rate = (
        resolved_pricing.cached_input_eur_per_million_tokens
        if resolved_pricing.cached_input_eur_per_million_tokens is not None
        else input_rate
    )
    reasoning_rate = (
        resolved_pricing.reasoning_eur_per_million_tokens
        if resolved_pricing.reasoning_eur_per_million_tokens is not None
        else output_rate
    )

    if cached_input_eur_per_million_tokens is not None:
        cached_rate = _coerce_decimal(
            cached_input_eur_per_million_tokens,
            label="cached-input EUR rate",
        )
    if reasoning_eur_per_million_tokens is not None:
        reasoning_rate = _coerce_decimal(
            reasoning_eur_per_million_tokens,
            label="reasoning-output EUR rate",
        )
    if cached_input_rate_eur_per_million is not None:
        cached_rate = _coerce_decimal(
            cached_input_rate_eur_per_million,
            label="cached-input EUR rate",
        )
    if reasoning_rate_eur_per_million is not None:
        reasoning_rate = _coerce_decimal(
            reasoning_rate_eur_per_million,
            label="reasoning-output EUR rate",
        )

    def amount(tokens: int, rate: Decimal) -> Decimal:
        with localcontext() as context:
            context.prec = 80
            return _round_money(
                Decimal(tokens) * rate / TOKENS_PER_PRICE_UNIT
            )

    uncached_input_cost = amount(
        resolved_usage.uncached_input_tokens, input_rate
    )
    cached_input_cost = amount(
        resolved_usage.cached_input_tokens, cached_rate
    )
    non_reasoning_output_cost = amount(
        resolved_usage.non_reasoning_output_tokens, output_rate
    )
    reasoning_output_cost = amount(
        resolved_usage.reasoning_tokens, reasoning_rate
    )
    input_cost = _round_money(uncached_input_cost + cached_input_cost)
    output_cost = _round_money(
        non_reasoning_output_cost + reasoning_output_cost
    )

    return CostBreakdown(
        model=resolved_pricing.model,
        input_tokens=resolved_usage.input_tokens,
        output_tokens=resolved_usage.output_tokens,
        total_tokens=resolved_usage.total_tokens,
        cached_input_tokens=resolved_usage.cached_input_tokens,
        reasoning_tokens=resolved_usage.reasoning_tokens,
        input_cost_eur=input_cost,
        output_cost_eur=output_cost,
        total_cost_eur=_round_money(input_cost + output_cost),
        uncached_input_cost_eur=uncached_input_cost,
        cached_input_cost_eur=cached_input_cost,
        non_reasoning_output_cost_eur=non_reasoning_output_cost,
        reasoning_output_cost_eur=reasoning_output_cost,
        input_rate_eur_per_million_tokens=input_rate,
        output_rate_eur_per_million_tokens=output_rate,
        cached_input_rate_eur_per_million_tokens=cached_rate,
        reasoning_rate_eur_per_million_tokens=reasoning_rate,
        evaluation_count=1,
        currency=EUR,
    )


def aggregate_token_usage(
    usages: TokenUsage | Mapping[str, Any] | Iterable[Any] | None,
) -> TokenUsage:
    """Sum usage records without double-counting subset token fields."""

    if usages is None:
        return TokenUsage()
    if isinstance(usages, (TokenUsage, Mapping, BaseModel)):
        items: list[Any] = [usages]
    else:
        try:
            items = list(usages)
        except TypeError as exc:
            raise PriceAccountingError(
                "token usages must be an iterable of usage records"
            ) from exc

    if isinstance(usages, Mapping):
        nested = _read_raw_alias(usages, ("usages", "records", "items"))
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            items = list(nested)

    input_total = 0
    output_total = 0
    cached_total = 0
    reasoning_total = 0

    for item in items:
        usage = _coerce_usage(item)
        input_total += usage.input_tokens
        output_total += usage.output_tokens
        cached_total += usage.cached_input_tokens
        reasoning_total += usage.reasoning_tokens
        if input_total > MAX_EXACT_TOKEN_COUNT or output_total > MAX_EXACT_TOKEN_COUNT:
            raise PriceAccountingError("aggregated token count exceeds exact limit")
        if cached_total > MAX_EXACT_TOKEN_COUNT or reasoning_total > MAX_EXACT_TOKEN_COUNT:
            raise PriceAccountingError("aggregated subset count exceeds exact limit")

    return TokenUsage(
        input_tokens=input_total,
        output_tokens=output_total,
        total_tokens=input_total + output_total,
        cached_input_tokens=cached_total,
        reasoning_tokens=reasoning_total,
    )


def _sequence_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
            memoryview,
            Mapping,
            TokenUsage,
            ModelPricing,
        ),
    ):
        return None
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return None


def _read_count(
    data: Mapping[Any, Any], *, default: int = 1
) -> int:
    value = _read_alias(
        data,
        _COST_COUNT_ALIASES,
        lambda item: _coerce_int(item, label="evaluation count"),
        label="evaluation count",
    )
    return default if value is None else value


def _record_parts(
    item: Any,
    *,
    default_pricing: Any,
) -> tuple[Any, Any, int]:
    if isinstance(item, Mapping):
        data = dict(item)
        raw_usage = _read_raw_alias(
            data,
            _USAGE_CONTAINER_ALIASES + ("token_usage",),
        )
        item_pricing = _read_raw_alias(
            data,
            ("pricing", "price_card", "priceCard", "rate_card", "model_pricing"),
        )
        if raw_usage is None:
            if not _looks_like_usage(data):
                raise PriceAccountingError(
                    "evaluation record does not contain token usage"
                )
            raw_usage = data
        if item_pricing is None:
            item_pricing = default_pricing
        return raw_usage, item_pricing, _read_count(data)

    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        if len(item) == 2 and (
            isinstance(item[0], (TokenUsage, Mapping, BaseModel))
            or hasattr(item[0], "input_tokens")
        ):
            return item[0], item[1], 1

    raw_usage = _read_raw_alias(item, ("usage", "token_usage")) if isinstance(
        item, Mapping
    ) else None
    if raw_usage is None:
        raw_usage = getattr(item, "usage", None)
    if raw_usage is None:
        raw_usage = getattr(item, "token_usage", None)
    if raw_usage is None and hasattr(item, "input_tokens"):
        raw_usage = item
    if raw_usage is None:
        raise PriceAccountingError(
            "evaluation record does not contain token usage"
        )

    item_pricing = getattr(item, "pricing", None)
    if item_pricing is None:
        item_pricing = default_pricing
    return raw_usage, item_pricing, _read_count(item)


def _scale_breakdown(
    breakdown: CostBreakdown,
    factor: int,
) -> CostBreakdown:
    if factor < 0:
        raise PriceAccountingError("evaluation multiplier cannot be negative")
    return CostBreakdown(
        model=breakdown.model,
        input_tokens=breakdown.input_tokens * factor,
        output_tokens=breakdown.output_tokens * factor,
        total_tokens=breakdown.total_tokens * factor,
        cached_input_tokens=breakdown.cached_input_tokens * factor,
        reasoning_tokens=breakdown.reasoning_tokens * factor,
        input_cost_eur=_round_money(breakdown.input_cost_eur * factor),
        output_cost_eur=_round_money(breakdown.output_cost_eur * factor),
        total_cost_eur=_round_money(breakdown.total_cost_eur * factor),
        uncached_input_cost_eur=_round_money(
            breakdown.uncached_input_cost_eur * factor
        ),
        cached_input_cost_eur=_round_money(
            breakdown.cached_input_cost_eur * factor
        ),
        non_reasoning_output_cost_eur=_round_money(
            breakdown.non_reasoning_output_cost_eur * factor
        ),
        reasoning_output_cost_eur=_round_money(
            breakdown.reasoning_output_cost_eur * factor
        ),
        input_rate_eur_per_million_tokens=(
            breakdown.input_rate_eur_per_million_tokens
        ),
        output_rate_eur_per_million_tokens=(
            breakdown.output_rate_eur_per_million_tokens
        ),
        cached_input_rate_eur_per_million_tokens=(
            breakdown.cached_input_rate_eur_per_million_tokens
        ),
        reasoning_rate_eur_per_million_tokens=(
            breakdown.reasoning_rate_eur_per_million_tokens
        ),
        evaluation_count=breakdown.evaluation_count * factor,
        currency=EUR,
    )


def _combine_breakdowns(parts: Sequence[CostBreakdown]) -> CostBreakdown:
    if not parts:
        return CostBreakdown(evaluation_count=0)

    models = {part.model for part in parts}
    model = next(iter(models)) if len(models) == 1 else "multiple"

    def same(field: str) -> Decimal | None:
        values = {getattr(part, field) for part in parts}
        if len(values) == 1:
            return next(iter(values))
        return None

    return CostBreakdown(
        model=model,
        input_tokens=sum(part.input_tokens for part in parts),
        output_tokens=sum(part.output_tokens for part in parts),
        total_tokens=sum(part.total_tokens for part in parts),
        cached_input_tokens=sum(part.cached_input_tokens for part in parts),
        reasoning_tokens=sum(part.reasoning_tokens for part in parts),
        input_cost_eur=_round_money(
            sum((part.input_cost_eur for part in parts), Decimal("0"))
        ),
        output_cost_eur=_round_money(
            sum((part.output_cost_eur for part in parts), Decimal("0"))
        ),
        total_cost_eur=_round_money(
            sum((part.total_cost_eur for part in parts), Decimal("0"))
        ),
        uncached_input_cost_eur=_round_money(
            sum(
                (part.uncached_input_cost_eur for part in parts),
                Decimal("0"),
            )
        ),
        cached_input_cost_eur=_round_money(
            sum(
                (part.cached_input_cost_eur for part in parts),
                Decimal("0"),
            )
        ),
        non_reasoning_output_cost_eur=_round_money(
            sum(
                (part.non_reasoning_output_cost_eur for part in parts),
                Decimal("0"),
            )
        ),
        reasoning_output_cost_eur=_round_money(
            sum(
                (part.reasoning_output_cost_eur for part in parts),
                Decimal("0"),
            )
        ),
        input_rate_eur_per_million_tokens=same(
            "input_rate_eur_per_million_tokens"
        ),
        output_rate_eur_per_million_tokens=same(
            "output_rate_eur_per_million_tokens"
        ),
        cached_input_rate_eur_per_million_tokens=same(
            "cached_input_rate_eur_per_million_tokens"
        ),
        reasoning_rate_eur_per_million_tokens=same(
            "reasoning_rate_eur_per_million_tokens"
        ),
        evaluation_count=sum(part.evaluation_count for part in parts),
        currency=EUR,
    )


def calculate_evaluation_cost(
    usage: TokenUsage | Mapping[str, Any] | Iterable[Any] | None = None,
    pricing: ModelPricing | Mapping[str, Any] | Iterable[Any] | None = None,
    evaluation_count: int = 1,
    *,
    evaluations: int | Iterable[Any] | None = None,
    count: int | None = None,
    n_evaluations: int | None = None,
    number_of_evaluations: int | None = None,
    token_usage: TokenUsage | Mapping[str, Any] | Iterable[Any] | None = None,
    usages: TokenUsage | Mapping[str, Any] | Iterable[Any] | None = None,
    model_pricing: ModelPricing | Mapping[str, Any] | None = None,
    price_card: ModelPricing | Mapping[str, Any] | None = None,
) -> CostBreakdown:
    """Calculate the aggregate EUR cost of one or more evaluations.

    ``usage`` may be a single usage record or an iterable of records. A
    record can be a usage object or a mapping containing ``usage`` and an
    optional ``pricing`` object. ``evaluation_count`` multiplies the
    resulting cost and token totals.
    """

    if usage is not None and token_usage is not None:
        raise PriceAccountingError("supply usage or token_usage, not both")
    if usage is not None and usages is not None:
        raise PriceAccountingError("supply usage or usages, not both")
    usage = usage if usage is not None else token_usage
    usage = usage if usage is not None else usages

    if pricing is not None and model_pricing is not None:
        raise PriceAccountingError("supply pricing or model_pricing, not both")
    if pricing is not None and price_card is not None:
        raise PriceAccountingError("supply pricing or price_card, not both")
    pricing = pricing if pricing is not None else model_pricing
    pricing = pricing if pricing is not None else price_card

    supplied_counts = [
        value
        for value in (
            count,
            n_evaluations,
            number_of_evaluations,
        )
        if value is not None
    ]
    if supplied_counts:
        parsed = [
            _coerce_int(value, label="evaluation count") for value in supplied_counts
        ]
        if any(value != parsed[0] for value in parsed[1:]):
            raise PriceAccountingError("conflicting evaluation counts were supplied")
        evaluation_count = parsed[0]
    evaluation_count = _coerce_int(
        evaluation_count, label="evaluation count"
    )

    record_source: list[Any] | None = None
    multiplier = evaluation_count

    if evaluations is not None and not isinstance(evaluations, int):
        record_source = _sequence_or_none(evaluations)
        if record_source is None:
            record_source = [evaluations]
    elif usage is not None:
        record_source = _sequence_or_none(usage)

    pricing_sequence = _sequence_or_none(pricing)
    default_pricing: Any = None if pricing_sequence is not None else pricing

    if record_source is not None:
        parts: list[CostBreakdown] = []
        for index, item in enumerate(record_source):
            item_default = default_pricing
            if pricing_sequence is not None:
                if index >= len(pricing_sequence):
                    raise PriceAccountingError(
                        "pricing sequence is shorter than the usage sequence"
                    )
                item_default = pricing_sequence[index]
            record_usage, record_pricing, record_count = _record_parts(
                item,
                default_pricing=item_default,
            )
            if record_pricing is None:
                raise PriceAccountingError(
                    "each evaluation record requires EUR pricing"
                )
            part = calculate_cost(record_usage, record_pricing)
            parts.append(
                _scale_breakdown(
                    part,
                    record_count * multiplier,
                )
            )
        return _combine_breakdowns(parts)

    if usage is None:
        raise PriceAccountingError("token usage is required")
    if pricing_sequence is not None:
        if len(pricing_sequence) != 1:
            raise PriceAccountingError(
                "a single usage record requires exactly one pricing record"
            )
        pricing = pricing_sequence[0]

    part = calculate_cost(usage, pricing)
    return _scale_breakdown(part, multiplier)


def format_eur(value: Decimal | int | float | str, *, places: int = 12) -> str:
    """Format a non-negative amount with an explicit euro sign."""

    places = _coerce_int(places, label="formatting places")
    if places > 18:
        raise PriceAccountingError("formatting places cannot exceed 18")
    amount = _coerce_decimal(value, label="EUR amount", maximum=None)
    quantum = Decimal("1").scaleb(-places)
    rounded = amount.quantize(quantum, rounding=ROUND_HALF_UP)
    text = format(rounded, f".{places}f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"€{text or '0'}"


def pricing_digest(
    pricing: ModelPricing | Mapping[str, Any],
) -> str:
    """Return a deterministic SHA-256 digest of a EUR rate card."""

    resolved = _coerce_pricing(pricing)
    payload = resolved.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def estimate_token_count(text: str | bytes) -> int:
    """Estimate tokens deterministically when a provider omits usage.

    This is intentionally an accounting fallback, not a provider billing
    claim. It uses a documented four-UTF-8-byte approximation.
    """

    if isinstance(text, (bytes, bytearray, memoryview)):
        try:
            decoded = bytes(text).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PriceAccountingError("text must be valid UTF-8") from exc
    elif isinstance(text, str):
        decoded = text
    else:
        raise PriceAccountingError("text must be a string or UTF-8 bytes")

    if not decoded:
        return 0
    byte_count = len(decoded.encode("utf-8"))
    return max(1, (byte_count + 3) // 4)


def count_tokens(text: str | bytes) -> int:
    """Compatibility alias for :func:`estimate_token_count`."""

    return estimate_token_count(text)


def usage_from_text(
    input_text: str | bytes = "",
    output_text: str | bytes = "",
) -> TokenUsage:
    """Create deterministic fallback usage from text lengths."""

    input_tokens = estimate_token_count(input_text)
    output_tokens = estimate_token_count(output_text)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


# Compatibility names for callers that use more descriptive type names.
PriceCard = ModelPricing
EvaluationCost = CostBreakdown
TokenAccounting = TokenUsage

__all__ = [
    "EUR",
    "EUR_ACCOUNTING_QUANTUM",
    "EvaluationCost",
    "MAX_EXACT_TOKEN_COUNT",
    "MAX_RATE_EUR_PER_MILLION",
    "ModelPricing",
    "PriceAccountingError",
    "PriceCard",
    "TOKENS_PER_PRICE_UNIT",
    "TokenAccounting",
    "TokenUsage",
    "aggregate_token_usage",
    "calculate_cost",
    "calculate_evaluation_cost",
    "count_tokens",
    "estimate_token_count",
    "extract_token_usage",
    "format_eur",
    "pricing_digest",
    "usage_from_text",
]