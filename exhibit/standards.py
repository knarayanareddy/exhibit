from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models import canonical_json_bytes, sha256_hex


DEFAULT_STANDARDS_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "standards" / "harmonized_standards.json"
)
CONSTITUTIONAL_RULE = "Not a conformity assessment. Counsel classifies."
_ENVIRONMENT_KEYS = (
    "EXHIBIT_STANDARDS_PATH",
    "EXHIBIT_HARMONIZED_STANDARDS_PATH",
    "HARMONIZED_STANDARDS_PATH",
)
_STRUCTURAL_FEATURE_KEYS = {
    "architecture",
    "attributes",
    "characteristics",
    "controls",
    "data",
    "evidence",
    "features",
    "metadata",
    "requirements",
    "system",
    "trace",
    "traces",
}


class CatalogError(ValueError):
    """Base error for standards-catalog operations."""


class CatalogNotFoundError(CatalogError, FileNotFoundError):
    """The configured local standards fixture cannot be located."""


class CatalogValidationError(CatalogError):
    """The standards fixture is syntactically or semantically invalid."""


class DuplicateStandardError(CatalogValidationError):
    """A catalog contains duplicate identifiers or designations."""


class UnknownStandardError(CatalogError, KeyError):
    """A requested standard identifier is absent from the catalog."""


class _DuplicateJSONKey(ValueError):
    pass


class HarmonizationStatus(str, Enum):
    TECHNICAL_REFERENCE = "technical-reference"
    DELEGATED_ACT_HarmonIZED = "delegated-act-harmonized"
    WITHDRAWN = "withdrawn"


class _CatalogModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
    )


def _normalise_nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def normalize_feature(value: Any) -> str:
    """Normalize a characteristic into a deterministic kebab-case token."""

    if value is None or isinstance(value, bool):
        return ""
    text = _normalise_nfkc(str(value)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or ""


def normalise_feature(value: Any) -> str:
    """British-spelling compatibility alias for :func:`normalize_feature`."""

    return normalize_feature(value)


def normalise_article_reference(value: Any) -> str:
    return normalize_article_reference(value)


def normalise_annex_section(value: Any) -> str:
    return normalize_annex_section(value)


def normalize_article_reference(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean values are not article references")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("article number must be positive")
        return f"Article {value}"
    if not isinstance(value, (str, int)):
        raise ValueError("article reference must be a string or integer")

    text = _normalise_nfkc(str(value)).casefold()
    article_match = re.search(r"\b(?:eu\s+ai\s+act\s+)?(?:article|art\.?)\s*\(?\s*(\d{1,3})\s*\)?", text)
    if article_match is None:
        number_match = re.fullmatch(r"\(?\s*(\d{1,3})\s*\)?", text)
        if number_match is None:
            raise ValueError(f"unsupported article reference: {value!r}")
        number = int(number_match.group(1))
    else:
        number = int(article_match.group(1))
    if number <= 0:
        raise ValueError("article number must be positive")
    return f"Article {number}"


def normalize_annex_section(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean values are not Annex IV section references")
    text = _normalise_nfkc(str(value)).casefold()
    text = text.replace("annex", "annex").replace("point", "point").replace("paragraph", "point")
    text = re.sub(r"\s+", " ", text).strip()

    match = re.fullmatch(
        r"(?:annex\s*)?(?:iv|4)\s*(?:[.,:]\s*)?(?:(?:point|section|no\.?|number)\s*)?\(?\s*(\d{1,2})(?:\.\d+)?\s*\)?",
        text,
    )
    if match is None:
        raise ValueError(f"unsupported Annex IV section reference: {value!r}")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError("Annex IV section number must be positive")
    return f"Annex IV.{number}"


def normalize_designation(value: Any) -> str:
    text = _normalise_nfkc(str(value))
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_standard_id(value: Any) -> str:
    designation = normalize_designation(value)
    normalized = normalize_feature(designation)
    if not normalized:
        raise ValueError("standard identifier must not be empty")
    return normalized


normalize_standard_identifier = normalize_standard_id


def _looks_like_article(value: str) -> bool:
    return bool(re.fullmatch(r"(?:eu\s+ai\s+act\s+)?(?:article|art\.?)\s*\(?\s*\d{1,3}[a-z]?(?:\([^)]*\))?\)?", value.casefold()))


def _article_number(value: str) -> int | None:
    try:
        reference = normalize_article_reference(value)
    except ValueError:
        return None
    match = re.search(r"\d+", reference)
    return int(match.group(0)) if match else None


def extract_feature_tokens(value: Any) -> tuple[str, ...]:
    """Extract deterministic feature phrases and article aliases from nested data."""

    tokens: set[str] = set()

    def add_text(text: Any, *, key: str = "") -> None:
        if text is None or isinstance(text, bool):
            return
        raw = _normalise_nfkc(str(text))
        if not raw:
            return
        normalized_key = normalize_feature(key)
        if _looks_like_article(raw) or normalized_key in {"article", "articles"}:
            try:
                article = normalize_article_reference(raw)
            except ValueError:
                if normalized_key in {"article", "articles"} and re.fullmatch(r"\d{1,3}", raw.strip()):
                    article = normalize_article_reference(int(raw.strip()))
                else:
                    article = ""
            if article:
                number = article.removeprefix("Article ")
                tokens.update({article, number, f"article-{number}"})
                return
        feature = normalize_feature(raw)
        if feature:
            tokens.add(feature)

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, BaseModel):
            walk(node.model_dump(mode="python", by_alias=True), key)
            return
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                normalized_key = normalize_feature(str(child_key))
                if child is True and normalized_key not in _STRUCTURAL_FEATURE_KEYS:
                    add_text(child_key, key=str(child_key))
                else:
                    walk(child, key=str(child_key))
            return
        if isinstance(node, str):
            add_text(node, key=key)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                walk(child, key=key)
            return
        if isinstance(node, (set, frozenset)):
            for child in sorted(node, key=str):
                walk(child, key=key)
            return
        if normalize_feature(key) in {"article", "articles"} and isinstance(node, int):
            add_text(node, key=key)

    walk(value)
    return tuple(sorted(tokens))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key '{key}'")
        result[key] = value
    return result


def _validate_non_empty_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("text must not be empty")
    return text


class SourceReference(_CatalogModel):
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    retrieved_on: date
    verification_required: bool = True

    @field_validator("publisher", "title", mode="before")
    @classmethod
    def validate_text(cls, value: Any) -> str:
        return _validate_non_empty_text(value)

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value: Any) -> str:
        text = _validate_non_empty_text(value)
        parsed = urlparse(text)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("source URL must be an absolute HTTP or HTTPS URL")
        return text

    @field_validator("retrieved_on", mode="before")
    @classmethod
    def validate_date(cls, value: Any) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))


class EvidenceRequirement(_CatalogModel):
    evidence_id: str = Field(min_length=1)
    article: str = Field(min_length=1)
    description: str = Field(min_length=1)
    characteristics: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_id", mode="before")
    @classmethod
    def validate_evidence_id(cls, value: Any) -> str:
        text = _validate_non_empty_text(value)
        return normalize_feature(text)

    @field_validator("article", mode="before")
    @classmethod
    def validate_article(cls, value: Any) -> str:
        return normalize_article_reference(value)

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, value: Any) -> str:
        return _validate_non_empty_text(value)

    @field_validator("characteristics", mode="before")
    @classmethod
    def validate_characteristics(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError("characteristics must be a sequence")
        normalised = tuple(
            dict.fromkeys(
                feature
                for item in value
                if (feature := normalize_feature(item))
            )
        )
        if not normalised:
            raise ValueError("characteristics must not be empty")
        return normalised


class StandardRecord(_CatalogModel):
    standard_id: str = Field(min_length=1)
    designation: str = Field(min_length=1)
    title: str = Field(min_length=1)
    issuing_body: str = Field(min_length=1)
    technical_committee: str | None = None
    edition: str = Field(min_length=1)
    harmonization_status: HarmonizationStatus
    official_journal_listing: bool = False
    articles: tuple[str, ...] = Field(min_length=1)
    annex_iv_sections: tuple[str, ...] = Field(default_factory=tuple)
    characteristics: tuple[str, ...] = Field(min_length=1)
    evidence_requirements: tuple[EvidenceRequirement, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    source: SourceReference

    @field_validator("standard_id", mode="before")
    @classmethod
    def validate_standard_id(cls, value: Any) -> str:
        return normalize_standard_id(value)

    @field_validator("designation", "title", "issuing_body", "technical_committee", "edition", mode="before")
    @classmethod
    def validate_text_fields(cls, value: Any) -> Any:
        if value is None:
            return None
        return _validate_non_empty_text(value)

    @field_validator("articles", mode="before")
    @classmethod
    def validate_articles(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError("articles must be a sequence")
        articles = tuple(dict.fromkeys(normalize_article_reference(item) for item in value))
        if not articles:
            raise ValueError("articles must not be empty")
        return tuple(sorted(articles, key=lambda article: int(article.split()[1])))

    @field_validator("annex_iv_sections", mode="before")
    @classmethod
    def validate_annex_sections(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError("annex_iv_sections must be a sequence")
        sections: list[str] = []
        for item in value:
            section = normalize_annex_section(item)
            if section not in sections:
                sections.append(section)
        return tuple(sorted(sections, key=lambda section: int(section.rsplit(".", 1)[1])))

    @field_validator("characteristics", mode="before")
    @classmethod
    def validate_characteristics(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError("characteristics must be a sequence")
        characteristics = tuple(
            dict.fromkeys(
                feature
                for item in value
                if (feature := normalize_feature(item))
            )
        )
        if not characteristics:
            raise ValueError("characteristics must not be empty")
        return characteristics

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            raise ValueError("limitations must be a sequence")
        limitations = tuple(
            dict.fromkeys(_validate_non_empty_text(item) for item in value)
        )
        if not limitations:
            raise ValueError("every standard must carry explicit limitations")
        return limitations

    @field_validator("evidence_requirements", mode="before")
    @classmethod
    def validate_requirements(cls, value: Any) -> tuple[EvidenceRequirement, ...]:
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError("evidence_requirements must be a sequence")
        return tuple(
            item if isinstance(item, EvidenceRequirement) else EvidenceRequirement.model_validate(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_evidence_scope(self) -> StandardRecord:
        evidence_ids = [requirement.evidence_id for requirement in self.evidence_requirements]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"duplicate evidence identifiers in '{self.standard_id}'")
        article_set = set(self.articles)
        for requirement in self.evidence_requirements:
            if requirement.article not in article_set:
                raise ValueError(
                    f"evidence '{requirement.evidence_id}' references {requirement.article}, "
                    f"which is absent from {self.designation}"
                )
        if self.official_journal_listing and self.harmonization_status != HarmonizationStatus.DELEGATED_ACT_HarmonIZED:
            raise ValueError(
                "an Official Journal listing requires delegated-act-harmonized status"
            )
        return self

    @property
    def id(self) -> str:
        return self.standard_id

    @property
    def references_official_journal(self) -> bool:
        return self.official_journal_listing

    def for_article(self, article: Any) -> bool:
        return normalize_article_reference(article) in self.articles

    def for_annex_section(self, section: Any) -> bool:
        return normalize_annex_section(section) in self.annex_iv_sections

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class CatalogDocument(_CatalogModel):
    catalog_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    constitutional_rule: str = Field(min_length=1)
    cache_policy: str = Field(min_length=1)
    last_reviewed: date
    standards: tuple[StandardRecord, ...] = Field(min_length=1)

    @field_validator("catalog_id", mode="before")
    @classmethod
    def validate_catalog_id(cls, value: Any) -> str:
        return normalize_feature(value)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> str:
        text = _validate_non_empty_text(value)
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", text):
            raise ValueError("schema_version must be a semantic version")
        return text

    @field_validator("title", "scope", mode="before")
    @classmethod
    def validate_text(cls, value: Any) -> str:
        return _validate_non_empty_text(value)

    @field_validator("constitutional_rule", mode="before")
    @classmethod
    def validate_constitutional_rule(cls, value: Any) -> str:
        text = _validate_non_empty_text(value)
        if text != CONSTITUTIONAL_RULE:
            raise ValueError(f"catalog constitutional rule must be '{CONSTITUTIONAL_RULE}'")
        return text

    @field_validator("cache_policy", mode="before")
    @classmethod
    def validate_cache_policy(cls, value: Any) -> str:
        text = _validate_non_empty_text(value)
        if text != "local-only":
            raise ValueError("this exhibit permits only a local-only standards cache")
        return text

    @field_validator("last_reviewed", mode="before")
    @classmethod
    def validate_review_date(cls, value: Any) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    @field_validator("standards", mode="before")
    @classmethod
    def validate_standards(cls, value: Any) -> tuple[StandardRecord, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray, Mapping)):
            raise ValueError("standards must be an array")
        records: list[StandardRecord] = []
        for item in value:
            if isinstance(item, StandardRecord):
                records.append(item)
            elif isinstance(item, Mapping):
                records.append(StandardRecord.model_validate(item))
            else:
                raise ValueError("each catalog standard must be an object")
        identifiers = [record.standard_id.casefold() for record in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("catalog contains duplicate standard identifiers")
        designations = [normalize_designation(record.designation).casefold() for record in records]
        if len(designations) != len(set(designations)):
            raise ValueError("catalog contains duplicate standard designations")
        return tuple(records)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


@dataclass(frozen=True, slots=True)
class StandardMatch:
    standard: StandardRecord
    score: float
    matched_characteristics: tuple[str, ...]
    missing_characteristics: tuple[str, ...]
    matched_articles: tuple[str, ...]
    matched_annex_sections: tuple[str, ...]
    matched_evidence_characteristics: tuple[str, ...]
    explanation: str

    @property
    def record(self) -> StandardRecord:
        return self.standard

    @property
    def standard_id(self) -> str:
        return self.standard.standard_id

    @property
    def designation(self) -> str:
        return self.standard.designation

    @property
    def matched_features(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.matched_characteristics)
                | set(self.matched_evidence_characteristics)
                | set(self.matched_articles)
                | set(self.matched_annex_sections)
            )
        )

    @property
    def coverage(self) -> float:
        return self.score

    @property
    def rationale(self) -> str:
        return self.explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard_id": self.standard.standard_id,
            "designation": self.standard.designation,
            "score": self.score,
            "matched_characteristics": list(self.matched_characteristics),
            "missing_characteristics": list(self.missing_characteristics),
            "matched_articles": list(self.matched_articles),
            "matched_annex_sections": list(self.matched_annex_sections),
            "matched_evidence_characteristics": list(self.matched_evidence_characteristics),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class StandardsCatalog:
    document: CatalogDocument
    source_name: str
    source_sha256: str

    def __iter__(self) -> Iterator[StandardRecord]:
        return iter(self.document.standards)

    def __len__(self) -> int:
        return len(self.document.standards)

    def __getitem__(self, identifier: str) -> StandardRecord:
        return self.get(identifier)

    @property
    def catalog_id(self) -> str:
        return self.document.catalog_id

    @property
    def version(self) -> str:
        return self.document.schema_version

    @property
    def standards(self) -> tuple[StandardRecord, ...]:
        return self.document.standards

    @property
    def title(self) -> str:
        return self.document.title

    @property
    def scope(self) -> str:
        return self.document.scope

    @property
    def last_reviewed(self) -> date:
        return self.document.last_reviewed

    def get(self, identifier: str) -> StandardRecord:
        query = normalize_designation(identifier).casefold()
        normalized_identifier = normalize_standard_id(identifier)
        for record in self.document.standards:
            if query in {
                record.standard_id.casefold(),
                record.designation.casefold(),
            } or normalized_identifier == record.standard_id:
                return record
        raise UnknownStandardError(f"unknown standard: {identifier}")

    def lookup(self, identifier: str) -> StandardRecord:
        return self.get(identifier)

    def find(self, identifier: str) -> StandardRecord | None:
        try:
            return self.get(identifier)
        except UnknownStandardError:
            return None

    def for_article(self, article: Any) -> tuple[StandardRecord, ...]:
        normalized = normalize_article_reference(article)
        return tuple(record for record in self.document.standards if normalized in record.articles)

    def for_articles(self, articles: Any) -> tuple[StandardRecord, ...]:
        if isinstance(articles, (str, int)):
            articles = [articles]
        normalized = tuple(dict.fromkeys(normalize_article_reference(article) for article in articles))
        return tuple(
            record
            for record in self.document.standards
            if any(article in record.articles for article in normalized)
        )

    def for_annex_iv_section(self, section: Any) -> tuple[StandardRecord, ...]:
        normalized = normalize_annex_section(section)
        return tuple(
            record
            for record in self.document.standards
            if normalized in record.annex_iv_sections
        )

    def for_characteristic(self, characteristic: Any) -> tuple[StandardRecord, ...]:
        normalized = normalize_feature(characteristic)
        return tuple(
            record
            for record in self.document.standards
            if any(normalized in normalize_feature(item) for item in record.characteristics)
        )

    def evidence_for_article(self, article: Any) -> tuple[EvidenceRequirement, ...]:
        normalized = normalize_article_reference(article)
        return tuple(
            requirement
            for record in self.document.standards
            for requirement in record.evidence_requirements
            if requirement.article == normalized
        )

    def match_architecture(
        self,
        architecture: Any,
        *,
        minimum_score: float = 0.0,
        limit: int | None = None,
    ) -> tuple[StandardMatch, ...]:
        """Rank standards deterministically against nested architecture features."""

        if architecture is None:
            return ()
        if isinstance(architecture, StandardRecord):
            architecture = architecture.model_dump(mode="python")
        elif isinstance(architecture, BaseModel):
            architecture = architecture.model_dump(mode="python", by_alias=True)

        feature_tokens = extract_feature_tokens(architecture)
        input_features = {
            token
            for token in feature_tokens
            if token not in {"12", "13", "14", "15", "50"}
            and not token.startswith("article-")
            and not token.startswith("annex-iv-")
        }
        input_articles = {
            f"Article {token}"
            for token in feature_tokens
            if token.isdigit()
        }
        input_annex = {
            token
            for token in feature_tokens
            if token.startswith("annex-iv-")
        }

        def collect_typed_references(node: Any, key: str = "") -> None:
            if isinstance(node, BaseModel):
                collect_typed_references(node.model_dump(mode="python", by_alias=True), key)
                return
            if isinstance(node, Mapping):
                for child_key, child in node.items():
                    collect_typed_references(child, str(child_key))
                return
            if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                for child in node:
                    collect_typed_references(child, key)
                return
            if not isinstance(node, str):
                return
            normalized_key = normalize_feature(key)
            if normalized_key in {"article", "articles"} or _looks_like_article(node):
                try:
                    input_articles.add(normalize_article_reference(node))
                except ValueError:
                    pass
            if "annex" in node.casefold():
                try:
                    input_annex.add(normalize_feature(normalize_annex_section(node)))
                except ValueError:
                    pass
            if normalized_key.startswith("annex") and node.strip().isdigit():
                input_annex.add(f"annex-iv-{int(node.strip())}")

        collect_typed_references(architecture)
        input_words = {
            word
            for feature in input_features
            for word in feature.split("-")
            if word and word not in {"and", "the", "for", "of", "to", "with"}
        }

        stop_words = {
            "ai",
            "and",
            "for",
            "information",
            "of",
            "system",
            "technical",
            "the",
            "to",
            "with",
        }

        def phrase_matches(target: str) -> bool:
            if not target:
                return False
            if target in input_features:
                return True
            target_words = {word for word in target.split("-") if word not in stop_words}
            if not target_words:
                return target in input_features
            overlap = len(target_words & input_words)
            return overlap == len(target_words) or overlap / len(target_words) >= 0.5

        ranked: list[StandardMatch] = []
        for index, record in enumerate(self.document.standards):
            matched_characteristics = tuple(
                characteristic
                for characteristic in record.characteristics
                if phrase_matches(characteristic)
            )
            missing_characteristics = tuple(
                characteristic
                for characteristic in record.characteristics
                if characteristic not in matched_characteristics
            )

            evidence_characteristics = tuple(
                dict.fromkeys(
                    characteristic
                    for requirement in record.evidence_requirements
                    for characteristic in requirement.characteristics
                )
            )
            matched_evidence = tuple(
                characteristic
                for characteristic in evidence_characteristics
                if phrase_matches(characteristic)
            )

            matched_articles = tuple(article for article in record.articles if article in input_articles)
            matched_annex = tuple(
                section
                for section in record.annex_iv_sections
                if normalize_feature(section) in input_annex
            )

            weighted_score = 0.0
            weight_total = 0.0
            non_article_input = bool(input_features)
            if non_article_input and record.characteristics:
                characteristic_ratio = len(matched_characteristics) / len(record.characteristics)
                evidence_ratio = (
                    len(matched_evidence) / len(evidence_characteristics)
                    if evidence_characteristics
                    else characteristic_ratio
                )
                weighted_score += 0.70 * characteristic_ratio + 0.15 * evidence_ratio
                weight_total += 0.85
            if input_articles and record.articles:
                weighted_score += 0.25 * (len(matched_articles) / len(record.articles))
                weight_total += 0.25
            if input_annex and record.annex_iv_sections:
                weighted_score += 0.15 * (len(matched_annex) / len(record.annex_iv_sections))
                weight_total += 0.15
            if weight_total == 0:
                score = 0.0
            else:
                score = round(weighted_score / weight_total, 6)

            characteristic_text = (
                f"characteristics {len(matched_characteristics)}/{len(record.characteristics)}"
                if record.characteristics
                else "no characteristic checklist"
            )
            article_text = (
                ", ".join(matched_articles) if matched_articles else "no matching article"
            )
            annex_text = (
                ", ".join(matched_annex) if matched_annex else "no matching Annex IV section"
            )
            explanation = (
                f"{score:.3f} deterministic feature score; {characteristic_text}; "
                f"articles: {article_text}; Annex IV: {annex_text}; "
                "association is technical documentation support, not a conformity determination."
            )
            match = StandardMatch(
                standard=record,
                score=score,
                matched_characteristics=matched_characteristics,
                missing_characteristics=missing_characteristics,
                matched_articles=matched_articles,
                matched_annex_sections=matched_annex,
                matched_evidence_characteristics=matched_evidence,
                explanation=explanation,
            )
            if score > minimum_score or (minimum_score == 0 and score > 0):
                ranked.append((match, index))  # type: ignore[arg-type]

        ranked.sort(key=lambda item: (-item[0].score, item[1]))  # type: ignore[index,arg-type]
        matches = tuple(item[0] for item in ranked)  # type: ignore[misc]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            return matches[:limit]
        return matches

    def lookup_architecture(self, architecture: Any, **kwargs: Any) -> tuple[StandardMatch, ...]:
        return self.match_architecture(architecture, **kwargs)

    def match(self, architecture: Any, **kwargs: Any) -> tuple[StandardMatch, ...]:
        return self.match_architecture(architecture, **kwargs)

    def best_matches(self, architecture: Any, count: int = 3) -> tuple[StandardMatch, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return self.match_architecture(architecture, limit=count)

    def to_document_dict(self) -> dict[str, Any]:
        return self.document.to_dict()

    def canonical_sha256(self) -> str:
        return sha256_hex(self.document.model_dump(mode="python"))


def resolve_standards_path(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve explicit, environment, then repository-local catalog precedence."""

    if path is not None:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise CatalogNotFoundError(f"standards catalog does not exist: {resolved}")
        return resolved.resolve()

    environment = os.environ if environ is None else environ
    for key in _ENVIRONMENT_KEYS:
        configured = environment.get(key)
        if configured:
            resolved = Path(configured).expanduser()
            if not resolved.is_file():
                raise CatalogNotFoundError(
                    f"standards catalog configured by {key} does not exist: {resolved}"
                )
            return resolved.resolve()

    if not DEFAULT_STANDARDS_PATH.is_file():
        raise CatalogNotFoundError(
            f"default standards catalog does not exist: {DEFAULT_STANDARDS_PATH}"
        )
    return DEFAULT_STANDARDS_PATH.resolve()


def validate_catalog_payload(payload: Any) -> CatalogDocument:
    if not isinstance(payload, Mapping):
        raise CatalogValidationError("standards catalog root must be a JSON object")
    identifiers: set[str] = set()
    designations: set[str] = set()
    raw_standards = payload.get("standards")
    if not isinstance(raw_standards, Sequence) or isinstance(raw_standards, (str, bytes, bytearray)):
        raise CatalogValidationError("standards catalog must contain a standards array")
    for index, record in enumerate(raw_standards):
        if not isinstance(record, Mapping):
            raise CatalogValidationError(f"standard at index {index} must be an object")
        try:
            standard_id = normalize_standard_id(record.get("standard_id"))
            designation = normalize_designation(record.get("designation")).casefold()
        except (TypeError, ValueError) as exc:
            raise CatalogValidationError(f"standard at index {index} has invalid identity: {exc}") from exc
        if standard_id in identifiers:
            raise DuplicateStandardError(f"duplicate standard identifier: {standard_id}")
        if designation in designations:
            raise DuplicateStandardError(f"duplicate standard designation: {designation}")
        identifiers.add(standard_id)
        designations.add(designation)

    try:
        return CatalogDocument.model_validate(payload)
    except ValidationError as exc:
        raise CatalogValidationError(f"invalid standards catalog: {exc}") from exc


def load_standards_catalog(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> StandardsCatalog:
    resolved = resolve_standards_path(path, environ=environ)
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise CatalogNotFoundError(f"unable to read standards catalog: {resolved}") from exc

    try:
        text = raw.decode("utf-8-sig")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except UnicodeDecodeError as exc:
        raise CatalogValidationError(f"standards catalog is not UTF-8: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogValidationError(
            f"standards catalog is not valid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except _DuplicateJSONKey as exc:
        raise CatalogValidationError(f"standards catalog contains {exc}") from exc

    document = validate_catalog_payload(payload)
    return StandardsCatalog(
        document=document,
        source_name=resolved.name,
        source_sha256=sha256_hex(raw),
    )


def load_harmonized_standards(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> StandardsCatalog:
    return load_standards_catalog(path, environ=environ)


def catalog_digest(catalog: StandardsCatalog) -> str:
    return sha256_hex(canonical_json_bytes(catalog.document.model_dump(mode="python")))


__all__ = [
    "CONSTITUTIONAL_RULE",
    "CatalogDocument",
    "CatalogError",
    "CatalogNotFoundError",
    "CatalogValidationError",
    "DuplicateStandardError",
    "EvidenceRequirement",
    "HarmonizationStatus",
    "SourceReference",
    "StandardMatch",
    "StandardRecord",
    "StandardsCatalog",
    "UnknownStandardError",
    "catalog_digest",
    "extract_feature_tokens",
    "load_harmonized_standards",
    "load_standards_catalog",
    "normalise_annex_section",
    "normalise_article_reference",
    "normalise_feature",
    "normalize_annex_section",
    "normalize_article_reference",
    "normalize_designation",
    "normalize_feature",
    "normalize_standard_id",
    "resolve_standards_path",
    "validate_catalog_payload",
]