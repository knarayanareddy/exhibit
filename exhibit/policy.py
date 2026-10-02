from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import (
    CONSTITUTIONAL_RULE,
    Trace,
    canonical_json_bytes,
    compute_trace_sha256,
    sha256_hex,
    verify_human_review_bytes,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()

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
_AUTOMATED_ACTORS = {
    "agent",
    "ai",
    "automated",
    "bot",
    "compiler",
    "engine",
    "evaluator",
    "machine",
    "model",
    "robot",
    "service",
    "service-account",
    "software",
    "system",
}
_AUTOMATED_STAMP_MARKERS = {
    "approved",
    "certification",
    "certified",
    "compliance",
    "compliant",
    "conformant",
    "conformity",
    "legal-approval",
    "legally-compliant",
}


class PolicyError(RuntimeError):
    """Base class for deterministic governance-policy failures."""


class AutomatedComplianceStampProhibited(PolicyError):
    """Raised when software attempts to issue a legal or compliance claim."""


class HumanReviewRequired(PolicyError):
    """Raised when an export lacks an approved, signed human review."""


class InvalidHumanReviewError(PolicyError):
    """Raised when a purported human review is not an approved human record."""


class PolicyBindingError(PolicyError):
    """Raised when trace, evaluation, job, or review evidence is mismatched."""


class ExportNotAuthorized(PolicyError):
    """Raised when one or more mandatory export gates fail."""

    def __init__(self, decision: GovernanceDecision):
        self.decision = decision
        codes = ", ".join(decision.violation_codes) or "policy-denied"
        super().__init__(f"export denied by governance policy: {codes}")


class PolicyNodeStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


class PolicyRequestAction(str, Enum):
    EXPORT = "export"
    COMPILE = "compile"
    PREVIEW = "preview"


@dataclass(frozen=True, slots=True)
class PolicyNode:
    """One deterministic node in the immutable governance decision graph."""

    node_id: str
    description: str
    required: bool
    status: PolicyNodeStatus
    detail: str
    evidence_digest: str

    @property
    def passed(self) -> bool:
        return self.status is PolicyNodeStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status is PolicyNodeStatus.FAIL

    @property
    def pending(self) -> bool:
        return self.status is PolicyNodeStatus.PENDING

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "description": self.description,
            "required": self.required,
            "status": self.status.value,
            "detail": self.detail,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class GovernanceRequest:
    """Immutable input to the governance policy graph."""

    trace: Any
    evaluation: Any
    human_review: Any | None = None
    job_id: str | None = None
    public_key: bytes | None = None
    requested_action: PolicyRequestAction = PolicyRequestAction.EXPORT
    context: Mapping[str, Any] | None = None

    @property
    def review(self) -> Any | None:
        return self.human_review

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": _read(self.trace, "trace_id"),
            "trace_sha256": _read(self.trace, "trace_sha256"),
            "evaluation_id": _read(self.evaluation, "evaluation_id"),
            "evaluation_trace_sha256": _read(self.evaluation, "trace_sha256"),
            "job_id": self.job_id,
            "requested_action": PolicyRequestAction(
                self.requested_action
            ).value,
            "human_review_present": self.human_review is not None,
            "constitutional_rule": CONSTITUTIONAL_RULE,
        }


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """Immutable result of evaluating the governance policy graph."""

    authorized: bool
    requested_action: str
    policy_status: str
    human_review_required: bool
    human_review_verified: bool
    review_decision: str | None
    trace_id: str | None
    trace_sha256: str | None
    evaluation_id: str | None
    evaluation_trace_sha256: str | None
    review_sha256: str | None
    violation_codes: tuple[str, ...]
    nodes: tuple[PolicyNode, ...]
    constitutional_rule: str = CONSTITUTIONAL_RULE

    @property
    def passed(self) -> bool:
        return self.authorized

    @property
    def failed(self) -> bool:
        return not self.authorized

    @property
    def allowed(self) -> bool:
        return self.authorized

    @property
    def denied(self) -> bool:
        return not self.authorized

    @property
    def export_allowed(self) -> bool:
        return (
            self.authorized
            and self.requested_action == PolicyRequestAction.EXPORT.value
        )

    @property
    def compile_allowed(self) -> bool:
        return self.authorized and self.requested_action in {
            PolicyRequestAction.COMPILE.value,
            PolicyRequestAction.EXPORT.value,
        }

    @property
    def preview_allowed(self) -> bool:
        return (
            self.authorized
            and self.requested_action == PolicyRequestAction.PREVIEW.value
        )

    @property
    def can_export(self) -> bool:
        return self.export_allowed

    @property
    def policy_graph_sha256(self) -> str:
        return _safe_digest(
            {
                "nodes": [node.as_dict() for node in self.nodes],
                "violation_codes": self.violation_codes,
                "constitutional_rule": self.constitutional_rule,
            }
        )

    def __bool__(self) -> bool:
        return self.authorized

    def require_authorized(self) -> GovernanceDecision:
        if not self.authorized:
            raise ExportNotAuthorized(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "export_allowed": self.export_allowed,
            "compile_allowed": self.compile_allowed,
            "preview_allowed": self.preview_allowed,
            "requested_action": self.requested_action,
            "policy_status": self.policy_status,
            "human_review_required": self.human_review_required,
            "human_review_verified": self.human_review_verified,
            "review_decision": self.review_decision,
            "trace_id": self.trace_id,
            "trace_sha256": self.trace_sha256,
            "evaluation_id": self.evaluation_id,
            "evaluation_trace_sha256": self.evaluation_trace_sha256,
            "review_sha256": self.review_sha256,
            "violation_codes": list(self.violation_codes),
            "nodes": [node.as_dict() for node in self.nodes],
            "policy_graph_sha256": self.policy_graph_sha256,
            "constitutional_rule": self.constitutional_rule,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


def _read(source: Any, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    for name in names:
        try:
            return getattr(source, name)
        except (AttributeError, TypeError):
            continue
    return default


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


def _normalised_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _text(value).casefold()).strip("-")


def _identifier(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _sha256(value: Any) -> str | None:
    text = _text(value).casefold()
    if text.startswith("0x"):
        text = text[2:]
    return text if _SHA256_RE.fullmatch(text) else None


def _strict_bool(value: Any) -> bool | None:
    value = _scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().casefold()
        if normalised in {"true", "yes", "1"}:
            return True
        if normalised in {"false", "no", "0"}:
            return False
    return None


def _plain_value(source: Any) -> Any:
    if source is None or isinstance(source, (str, int, float, bool)):
        return source
    if isinstance(source, Enum):
        return source.value
    dump = getattr(source, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", by_alias=True)
        except TypeError:
            return dump()
    if isinstance(source, Mapping):
        return {str(key): _plain_value(value) for key, value in source.items()}
    if isinstance(source, (list, tuple, set, frozenset)):
        return [_plain_value(item) for item in source]
    return source


def _safe_digest(value: Any) -> str:
    try:
        return sha256_hex(canonical_json_bytes(value))
    except (TypeError, ValueError):
        return sha256_hex(
            canonical_json_bytes(
                {
                    "canonicalization_error": type(value).__name__,
                    "node_type": type(value).__name__,
                }
            )
        )


def _normalise_action(value: Any) -> str | None:
    text = _normalised_token(value)
    aliases = {
        "action-allow": "Action.allow",
        "allow": "Action.allow",
        "action-review": "Action.review",
        "review": "Action.review",
        "action-queue": "Action.queue",
        "queue": "Action.queue",
        "human-review": "Action.queue",
        "action-block": "Action.block",
        "block": "Action.block",
    }
    return aliases.get(text)


def _review_decision(value: Any) -> str:
    return _normalised_token(value)


def _is_approved_decision(value: Any) -> bool:
    return _review_decision(value) in _APPROVED_DECISIONS


def _is_rejected_decision(value: Any) -> bool:
    return _review_decision(value) in _REJECTED_DECISIONS


def _review_actor(value: Any) -> str:
    if isinstance(value, Mapping):
        value = _read(value, "actor_type", "type", "actor", default="")
    return _normalised_token(value)


def _review_value(review: Any, *names: str, default: Any = None) -> Any:
    value = _read(review, *names, default=_MISSING)
    if value is not _MISSING:
        return value
    scope = _read(review, "scope", "context", "binding")
    if scope is not None:
        value = _read(scope, *names, default=_MISSING)
        if value is not _MISSING:
            return value
    return default


def _public_key_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return None


def _review_signing_message(review: Any) -> bytes:
    signing_bytes = _read(review, "signing_bytes", "signed_bytes")
    if callable(signing_bytes):
        try:
            value = signing_bytes()
        except (TypeError, ValueError):
            value = None
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)

    plain = _plain_value(review)
    if isinstance(plain, Mapping):
        material = {
            str(key): value
            for key, value in plain.items()
            if str(key)
            not in {
                "signature",
                "public_key",
                "verification",
                "verified",
                "verification_calls",
            }
        }
    else:
        material = plain
    return canonical_json_bytes(material)


def _invoke_named_verifier(
    function: Any,
    *,
    review: Any,
    public_key: bytes,
    message: bytes,
    signature: str,
) -> bool:
    values = {
        "public": public_key,
        "key": public_key,
        "publickey": public_key,
        "message": message,
        "payload": message,
        "bytes": message,
        "signedbytes": message,
        "signingbytes": message,
        "signature": signature,
        "sig": signature,
        "review": review,
        "humanreview": review,
        "signoff": review,
        "model": review,
        "record": review,
    }

    try:
        signature_inspection = inspect.signature(function)
    except (TypeError, ValueError):
        for arguments in (
            (public_key, message, signature),
            (review, public_key),
            (public_key, signature, message),
            (review,),
        ):
            try:
                return bool(function(*arguments))
            except (TypeError, ValueError):
                continue
        return False

    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    unresolved: list[str] = []

    for parameter in signature_inspection.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        normalised = re.sub(r"[^a-z0-9]", "", parameter.name.casefold())
        value = values.get(normalised, _MISSING)
        if value is _MISSING:
            unresolved.append(parameter.name)
            value = review
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[parameter.name] = value

    if unresolved and not any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature_inspection.parameters.values()
    ):
        return False

    try:
        if any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in signature_inspection.parameters.values()
        ):
            return bool(function(*positional, public_key, message, signature))
        return bool(function(*positional, **keywords))
    except (TypeError, ValueError):
        return False


def _verify_review_signature(
    review: Any,
    public_key_value: Any,
) -> tuple[bool, str]:
    public_key = _public_key_bytes(public_key_value)
    if public_key is None:
        return False, "a trusted public key is required for human-review verification"

    signature_value = _review_value(review, "signature", "signed_signature")
    signature = _text(signature_value)
    if not signature:
        return False, "human-review signature is missing"

    method = _read(review, "verify_signature")
    if callable(method):
        try:
            result = method(
                public_key,
                _review_signing_message(review),
                signature,
            )
            return bool(result), "model verifier" if result else "model verifier rejected signature"
        except (TypeError, ValueError):
            return False, "model verifier could not validate signature"

    method = _read(review, "verify")
    if callable(method):
        try:
            parameters = inspect.signature(method).parameters
            if len(parameters) == 0:
                result = method()
            elif len(parameters) == 1:
                result = method(public_key)
            else:
                result = method(public_key, signature)
            return bool(result), "model verifier" if result else "model verifier rejected signature"
        except (TypeError, ValueError):
            return False, "model verifier could not validate signature"

    try:
        verified = _invoke_named_verifier(
            verify_human_review_bytes,
            review=review,
            public_key=public_key,
            message=_review_signing_message(review),
            signature=signature,
        )
    except (KeyError, TypeError, ValueError):
        verified = False

    return verified, "cryptographic verifier" if verified else "signature verification failed"


def _contains_automated_stamp_marker(value: Any) -> bool:
    text = _normalised_token(value)
    if not text:
        return False
    tokens = set(text.split("-"))
    return bool(tokens & _AUTOMATED_STAMP_MARKERS) or any(
        marker in text for marker in _AUTOMATED_STAMP_MARKERS
    )


def _actor_is_automated(value: Any) -> bool:
    normalised = _normalised_token(value)
    return not normalised or normalised in _AUTOMATED_ACTORS


def _append_unique(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def _node(
    *,
    node_id: str,
    description: str,
    required: bool,
    issues: list[str],
    pass_detail: str,
    evidence: Any,
) -> PolicyNode:
    unique_issues = tuple(dict.fromkeys(issues))
    status = PolicyNodeStatus.FAIL if unique_issues else PolicyNodeStatus.PASS
    detail = "; ".join(unique_issues) if unique_issues else pass_detail
    return PolicyNode(
        node_id=node_id,
        description=description,
        required=required,
        status=status,
        detail=detail,
        evidence_digest=_safe_digest(
            {
                "node_id": node_id,
                "required": required,
                "issues": list(unique_issues),
                "evidence": evidence,
                "constitutional_rule": CONSTITUTIONAL_RULE,
            }
        ),
    )


class GovernancePolicy:
    """Pure-Python deterministic governance policy for evidence exports."""

    constitutional_rule = CONSTITUTIONAL_RULE

    def __init__(self, *, constitutional_rule: str = CONSTITUTIONAL_RULE) -> None:
        if constitutional_rule != CONSTITUTIONAL_RULE:
            raise PolicyError("the constitutional rule cannot be modified")

    def authorize_export(
        self,
        trace: Any | GovernanceRequest,
        evaluation: Any | None = None,
        human_review: Any | None = None,
        job_id: str | None = None,
        public_key: bytes | str | None = None,
        *,
        review: Any | None = None,
        requested_action: PolicyRequestAction | str = PolicyRequestAction.EXPORT,
    ) -> GovernanceDecision:
        """Evaluate the mandatory export gates without issuing a compliance claim."""

        if isinstance(trace, GovernanceRequest):
            request = trace
            trace = request.trace
            evaluation = request.evaluation
            human_review = request.human_review
            job_id = request.job_id
            public_key = request.public_key
            requested_action = request.requested_action

        if evaluation is None:
            raise PolicyBindingError("evaluation evidence is required")

        if human_review is not None and review is not None:
            left = _safe_digest(_plain_value(human_review))
            right = _safe_digest(_plain_value(review))
            if left != right:
                raise PolicyBindingError("conflicting human-review records were supplied")
        if review is not None:
            human_review = review

        try:
            action_requested = PolicyRequestAction(requested_action)
        except ValueError as error:
            raise PolicyError(f"unsupported policy action: {requested_action!r}") from error

        trace_issues: list[str] = []
        evaluation_issues: list[str] = []
        review_issues: list[str] = []
        binding_issues: list[str] = []
        stamp_issues: list[str] = []

        trace_id = _identifier(_read(trace, "trace_id", "traceId", "traceID"))
        if trace_id is None:
            trace_issues.append("trace_identifier_missing")
        elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", trace_id):
            trace_issues.append("trace_identifier_invalid")

        declared_trace_hash = _sha256(
            _read(trace, "trace_sha256", "trace_digest", "sha256")
        )
        computed_trace_hash: str | None = None
        try:
            candidate_hash = _sha256(compute_trace_sha256(trace))
            if candidate_hash is not None:
                computed_trace_hash = candidate_hash
        except (AttributeError, TypeError, ValueError):
            computed_trace_hash = None

        if declared_trace_hash and computed_trace_hash:
            trace_hash = declared_trace_hash
            if declared_trace_hash != computed_trace_hash:
                trace_issues.append("trace_hash_mismatch")
        else:
            trace_hash = declared_trace_hash or computed_trace_hash
        if trace_hash is None:
            trace_issues.append("trace_hash_missing_or_invalid")

        evaluation_id = _identifier(
            _read(evaluation, "evaluation_id", "evaluationId")
        )
        if evaluation_id is None:
            evaluation_issues.append("evaluation_identifier_missing")

        evaluation_trace_id = _identifier(
            _read(evaluation, "trace_id", "traceId", "traceID")
        )
        if evaluation_trace_id is None:
            evaluation_issues.append("evaluation_trace_identifier_missing")
        elif trace_id is not None and evaluation_trace_id != trace_id:
            binding_issues.append("evaluation_trace_binding_mismatch")

        evaluation_trace_hash = _sha256(
            _read(
                evaluation,
                "trace_sha256",
                "trace_digest",
                "source_trace_sha256",
            )
        )
        if evaluation_trace_hash is None:
            evaluation_issues.append("evaluation_trace_hash_missing_or_invalid")
        elif trace_hash is not None and evaluation_trace_hash != trace_hash:
            binding_issues.append("evaluation_trace_hash_mismatch")

        schema_valid = _strict_bool(_read(evaluation, "schema_valid"))
        if schema_valid is not True:
            evaluation_issues.append("evaluation_schema_not_valid")

        injection_detected = _strict_bool(
            _read(evaluation, "injection_detected")
        )
        injection_caught = _strict_bool(_read(evaluation, "injection_caught"))
        if injection_detected is True and injection_caught is False:
            evaluation_issues.append("detected_injection_not_caught")
        if injection_caught is True and injection_detected is False:
            evaluation_issues.append("caught_injection_not_detected")

        evaluation_action = _normalise_action(
            _read(evaluation, "policy_action", "policy_status", "recommended_action")
        )
        if evaluation_action is None:
            evaluation_issues.append("evaluation_policy_action_missing_or_invalid")
            evaluation_action = "Action.queue"
        elif evaluation_action == "Action.block":
            evaluation_issues.append("evaluation_requires_block")

        for source, label in (
            (trace, "trace"),
            (evaluation, "evaluation"),
        ):
            for field_name in (
                "automated_stamp",
                "automated_compliance_stamp",
                "compliance_stamp",
                "conformity_stamp",
            ):
                stamp = _read(source, field_name)
                if stamp is None:
                    continue
                if isinstance(stamp, Mapping):
                    claim = _read(stamp, "claim", "value", "label", "status")
                    actor = _read(stamp, "actor_type", "actor", "issuer")
                else:
                    claim = stamp
                    actor = "system"
                if _actor_is_automated(actor) and _contains_automated_stamp_marker(claim):
                    stamp_issues.append(f"automated_compliance_stamp_prohibited:{label}")

        review_job_id = _identifier(_review_value(review, "job_id", "case_id"))
        trace_job_id = _identifier(_read(trace, "job_id"))
        evaluation_job_id = _identifier(_read(evaluation, "job_id"))
        effective_job_id = (
            _identifier(job_id)
            or trace_job_id
            or evaluation_job_id
            or review_job_id
        )

        human_review_required = action_requested is not PolicyRequestAction.PREVIEW
        review_decision_value: str | None = None
        review_sha256: str | None = None
        human_review_verified = False

        if human_review_required:
            if human_review is None:
                review_issues.append("signed_human_review_required")
            else:
                review_sha256 = _safe_digest(_plain_value(human_review))
                actor = _review_actor(
                    _review_value(
                        human_review,
                        "actor_type",
                        "actor",
                        "reviewer_type",
                        "signer_type",
                    )
                )
                if actor != "human":
                    review_issues.append("human_reviewer_required")

                reviewer_id = _identifier(
                    _review_value(
                        human_review,
                        "reviewer_id",
                        "reviewer_name",
                        "actor_id",
                        "signer_id",
                    )
                )
                if reviewer_id is None:
                    review_issues.append("human_reviewer_identifier_missing")

                decision_value = _review_value(
                    human_review,
                    "decision",
                    "review_decision",
                    "disposition",
                )
                review_decision_value = _review_decision(decision_value) or None
                if decision_value in (None, ""):
                    review_issues.append("human_review_decision_missing")
                elif _is_rejected_decision(decision_value):
                    review_issues.append("human_review_rejected")
                elif not _is_approved_decision(decision_value):
                    review_issues.append("human_review_decision_invalid")

                signature_verified, _reason = _verify_review_signature(
                    human_review,
                    public_key,
                )
                if not signature_verified:
                    review_issues.append("human_review_signature_invalid")

                human_review_verified = not review_issues
        elif human_review is not None:
            review_sha256 = _safe_digest(_plain_value(human_review))
            decision_value = _review_value(
                human_review,
                "decision",
                "review_decision",
                "disposition",
            )
            review_decision_value = _review_decision(decision_value) or None
            actor = _review_actor(
                _review_value(human_review, "actor_type", "actor", "reviewer_type")
            )
            signature_verified, _reason = _verify_review_signature(
                human_review,
                public_key,
            )
            human_review_verified = (
                actor == "human"
                and signature_verified
                and _is_approved_decision(decision_value)
            )

        if human_review_required:
            review_trace_id = _identifier(
                _review_value(human_review, "trace_id", "traceId")
            )
            review_trace_hash = _sha256(
                _review_value(human_review, "trace_sha256", "trace_digest")
            )
            review_evaluation_id = _identifier(
                _review_value(human_review, "evaluation_id", "evaluationId")
            )

            if review_trace_id is None:
                binding_issues.append("review_trace_identifier_missing")
            elif trace_id is not None and review_trace_id != trace_id:
                binding_issues.append("review_trace_binding_mismatch")

            if review_trace_hash is None:
                binding_issues.append("review_trace_hash_missing_or_invalid")
            elif trace_hash is not None and review_trace_hash != trace_hash:
                binding_issues.append("review_trace_hash_mismatch")

            if review_evaluation_id is None:
                binding_issues.append("review_evaluation_identifier_missing")
            elif evaluation_id is not None and review_evaluation_id != evaluation_id:
                binding_issues.append("review_evaluation_binding_mismatch")

            if effective_job_id is None:
                binding_issues.append("job_identifier_missing")
            if review_job_id is None:
                binding_issues.append("review_job_identifier_missing")
            elif effective_job_id is not None and review_job_id != effective_job_id:
                binding_issues.append("review_job_binding_mismatch")

            if trace_job_id is not None and trace_job_id != effective_job_id:
                binding_issues.append("trace_job_binding_mismatch")
            if evaluation_job_id is not None and evaluation_job_id != effective_job_id:
                binding_issues.append("evaluation_job_binding_mismatch")
        elif binding_issues:
            binding_issues.append("preview_does_not_override_evidence_binding")

        violations: list[str] = []
        for issue in (
            *trace_issues,
            *evaluation_issues,
            *review_issues,
            *binding_issues,
            *stamp_issues,
        ):
            _append_unique(violations, issue)

        structural_failure = bool(trace_issues or evaluation_issues or binding_issues or stamp_issues)
        rejected = "human_review_rejected" in review_issues
        if structural_failure or rejected:
            policy_status = "Action.block"
        elif review_issues:
            policy_status = "Action.queue"
        else:
            policy_status = evaluation_action

        nodes = (
            _node(
                node_id="constitutional_rule",
                description="Preserve the non-conformity constitutional rule.",
                required=True,
                issues=[],
                pass_detail=CONSTITUTIONAL_RULE,
                evidence=CONSTITUTIONAL_RULE,
            ),
            _node(
                node_id="trace_evidence",
                description="Bind the export to an identified, hashed trace.",
                required=True,
                issues=trace_issues,
                pass_detail="trace identity and SHA-256 binding are present",
                evidence={
                    "trace_id": trace_id,
                    "trace_sha256": trace_hash,
                },
            ),
            _node(
                node_id="evaluation_evidence",
                description="Require a schema-valid, trace-bound evaluation.",
                required=True,
                issues=evaluation_issues,
                pass_detail="evaluation is valid and cryptographically bound",
                evidence={
                    "evaluation_id": evaluation_id,
                    "trace_sha256": evaluation_trace_hash,
                    "schema_valid": schema_valid,
                    "policy_action": evaluation_action,
                },
            ),
            _node(
                node_id="signed_human_review",
                description=(
                    "Require an approved human review with a valid signature."
                    if human_review_required
                    else "Human review is not mandatory for a non-export preview."
                ),
                required=human_review_required,
                issues=review_issues,
                pass_detail=(
                    "approved human review signature verified"
                    if human_review_required
                    else "preview does not create an export artifact"
                ),
                evidence={
                    "review_sha256": review_sha256,
                    "actor_type": _review_actor(
                        _review_value(human_review, "actor_type", "actor")
                    ),
                    "decision": review_decision_value,
                },
            ),
            _node(
                node_id="evidence_binding",
                description="Bind the trace, evaluation, review, and job identifiers.",
                required=True,
                issues=binding_issues,
                pass_detail="all evidence identifiers and hashes are bound",
                evidence={
                    "job_id": effective_job_id,
                    "trace_id": trace_id,
                    "trace_sha256": trace_hash,
                    "evaluation_id": evaluation_id,
                    "evaluation_trace_sha256": evaluation_trace_hash,
                    "review_job_id": review_job_id,
                },
            ),
            _node(
                node_id="automated_stamp_prohibition",
                description="Prohibit software-issued compliance or conformity stamps.",
                required=True,
                issues=stamp_issues,
                pass_detail="no automated compliance stamp was asserted",
                evidence={
                    "constitutional_rule": CONSTITUTIONAL_RULE,
                    "issuer": "space-bunny-alpha-compiler",
                },
            ),
        )

        authorized = not violations
        return GovernanceDecision(
            authorized=authorized,
            requested_action=action_requested.value,
            policy_status=policy_status,
            human_review_required=human_review_required,
            human_review_verified=human_review_verified,
            review_decision=review_decision_value,
            trace_id=trace_id,
            trace_sha256=trace_hash,
            evaluation_id=evaluation_id,
            evaluation_trace_sha256=evaluation_trace_hash,
            review_sha256=review_sha256,
            violation_codes=tuple(violations),
            nodes=nodes,
        )

    def authorize(
        self,
        trace: Any | GovernanceRequest,
        evaluation: Any | None = None,
        human_review: Any | None = None,
        job_id: str | None = None,
        public_key: bytes | str | None = None,
        **kwargs: Any,
    ) -> GovernanceDecision:
        return self.authorize_export(
            trace,
            evaluation,
            human_review,
            job_id,
            public_key,
            **kwargs,
        )

    def evaluate(
        self,
        request_or_trace: GovernanceRequest | Any,
        evaluation: Any | None = None,
        human_review: Any | None = None,
        job_id: str | None = None,
        public_key: bytes | str | None = None,
        **kwargs: Any,
    ) -> GovernanceDecision:
        return self.authorize_export(
            request_or_trace,
            evaluation,
            human_review,
            job_id,
            public_key,
            **kwargs,
        )

    def evaluate_export(self, *args: Any, **kwargs: Any) -> GovernanceDecision:
        return self.authorize_export(*args, **kwargs)

    def require_authorization(
        self,
        trace: Any | GovernanceRequest,
        evaluation: Any | None = None,
        human_review: Any | None = None,
        job_id: str | None = None,
        public_key: bytes | str | None = None,
        **kwargs: Any,
    ) -> GovernanceDecision:
        decision = self.authorize_export(
            trace,
            evaluation,
            human_review,
            job_id,
            public_key,
            **kwargs,
        )
        if decision.authorized:
            return decision

        if "signed_human_review_required" in decision.violation_codes:
            raise HumanReviewRequired(
                "an approved, cryptographically signed human review is required"
            )

        invalid_review_codes = {
            "human_reviewer_required",
            "human_reviewer_identifier_missing",
            "human_review_decision_missing",
            "human_review_decision_invalid",
            "human_review_rejected",
            "human_review_signature_invalid",
        }
        if invalid_review_codes.intersection(decision.violation_codes):
            raise InvalidHumanReviewError(
                "human-review actor, decision, or signature validation failed"
            )

        binding_codes = {
            "evaluation_trace_binding_mismatch",
            "evaluation_trace_hash_mismatch",
            "evaluation_job_binding_mismatch",
            "review_trace_identifier_missing",
            "review_trace_binding_mismatch",
            "review_trace_hash_missing_or_invalid",
            "review_trace_hash_mismatch",
            "review_evaluation_identifier_missing",
            "review_evaluation_binding_mismatch",
            "job_identifier_missing",
            "review_job_identifier_missing",
            "review_job_binding_mismatch",
            "trace_job_binding_mismatch",
        }
        if binding_codes.intersection(decision.violation_codes):
            raise PolicyBindingError(
                "human-review evidence is not bound to the requested job, trace, and evaluation"
            )

        raise ExportNotAuthorized(decision)

    def check_automated_stamp(
        self,
        stamp: Any = None,
        actor_type: Any = None,
        *,
        actor: Any = None,
        claim: Any = None,
        label: Any = None,
    ) -> PolicyNode:
        """Check a proposed stamp without treating it as legal classification."""

        if isinstance(stamp, Mapping):
            claim = (
                claim
                if claim is not None
                else _read(stamp, "claim", "value", "label", "status", "stamp")
            )
            actor_type = (
                actor_type
                if actor_type is not None
                else _read(stamp, "actor_type", "actor", "issuer", "type")
            )
        elif claim is None:
            claim = label if label is not None else stamp

        if actor_type is None:
            actor_type = actor

        first = _normalised_token(stamp)
        second = _normalised_token(actor_type)
        if first in _AUTOMATED_ACTORS and second not in _AUTOMATED_ACTORS:
            actor_type = first
            claim = second if second else claim

        issuer = _text(actor_type)
        automated_issuer = _actor_is_automated(issuer)
        prohibited = automated_issuer and _contains_automated_stamp_marker(claim)
        issues = (
            ["automated_compliance_stamp_prohibited"] if prohibited else []
        )
        return _node(
            node_id="automated_stamp_prohibition",
            description="Prohibit software-issued legal or compliance claims.",
            required=True,
            issues=issues,
            pass_detail="the proposed stamp is not an automated compliance claim",
            evidence={
                "issuer": issuer or "unspecified-automated-actor",
                "claim": _text(claim),
            },
        )

    def authorize_stamp(
        self,
        stamp: Any = None,
        actor_type: Any = None,
        **kwargs: Any,
    ) -> PolicyNode:
        node = self.check_automated_stamp(stamp, actor_type, **kwargs)
        if node.failed:
            raise AutomatedComplianceStampProhibited(node.detail)
        return node

    def assert_no_automated_stamp(
        self,
        stamp: Any = None,
        actor_type: Any = None,
        **kwargs: Any,
    ) -> PolicyNode:
        return self.authorize_stamp(stamp, actor_type, **kwargs)

    def assert_automated_stamp_allowed(
        self,
        stamp: Any = None,
        actor_type: Any = None,
        **kwargs: Any,
    ) -> PolicyNode:
        return self.authorize_stamp(stamp, actor_type, **kwargs)

    def assert_stamp_allowed(
        self,
        stamp: Any = None,
        actor_type: Any = None,
        **kwargs: Any,
    ) -> PolicyNode:
        return self.authorize_stamp(stamp, actor_type, **kwargs)

    def validate_stamp(
        self,
        stamp: Any = None,
        actor_type: Any = None,
        **kwargs: Any,
    ) -> PolicyNode:
        return self.check_automated_stamp(stamp, actor_type, **kwargs)