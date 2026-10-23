"""One accountable inspection event for one physical part.

``preview`` never counts a product. ``finalize`` freezes and counts one verdict,
including RECHECK, and is idempotent. Callers should keep an incomplete session
open while asking the operator to flip the part. Cancellation is persisted but
does not count a completed inspection. A reset explicitly begins a new event.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ViewDecision:
    view: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None

    def __post_init__(self) -> None:
        if not self.view or not isinstance(self.view, str):
            raise ValueError("A view identity is required")
        if self.verdict not in {"PASS", "FAIL", "RECHECK"}:
            raise ValueError("Verdict must be PASS, FAIL or RECHECK")
        self.reasons = list(self.reasons)
        self.candidates = deepcopy(self.candidates)
        self.metrics = deepcopy(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy({"view": self.view, "verdict": self.verdict,
                         "reasons": self.reasons, "candidates": self.candidates,
                         "metrics": self.metrics, "model_version": self.model_version})


class SessionState(str, Enum):
    AWAITING_VIEWS = "AWAITING_VIEWS"
    WAITING_FOR_FLIP = "WAITING_FOR_FLIP"
    RECHECK_REQUIRED = "RECHECK_REQUIRED"
    READY_TO_FINALIZE = "READY_TO_FINALIZE"
    FAIL_LOCKED = "FAIL_LOCKED"
    RESULT_LOCKED = "RESULT_LOCKED"
    CANCELLED = "CANCELLED"


class InspectionSession:
    def __init__(self, part_id: str,
                 required_views: tuple[str, ...] = ("inner_face", "ribbed_face"), *,
                 inspection_mode: str = "automatic"):
        if not isinstance(part_id, str) or not part_id.strip():
            raise ValueError("A physical part ID is required")
        if not required_views or len(set(required_views)) != len(required_views):
            raise ValueError("Required views must be nonempty and unique")
        if any(not isinstance(v, str) or not v.strip() for v in required_views):
            raise ValueError("Every required view needs an identity")
        if inspection_mode not in {"automatic", "manual"}:
            raise ValueError("Inspection mode must be automatic or manual")
        self.part_id = part_id.strip()
        self.required_views = tuple(required_views)
        self.inspection_mode = inspection_mode
        self.reset_history: list[dict[str, Any]] = []
        self._begin()

    def _begin(self) -> None:
        self.event_id = str(uuid4())
        self.created_at = utc_now()
        self.completed_at: str | None = None
        self.decisions: dict[str, ViewDecision] = {}
        self.attempts: list[dict[str, Any]] = []
        self.final_decision: ViewDecision | None = None
        self.counted = False
        self.cancel_reason: str | None = None
        self.state = SessionState.AWAITING_VIEWS

    @property
    def missing_views(self) -> list[str]:
        return [v for v in self.required_views if v not in self.decisions]

    @property
    def invalid_views(self) -> list[str]:
        return [v for v, decision in self.decisions.items() if decision.verdict == "RECHECK"]

    @property
    def complete(self) -> bool:
        return not self.missing_views and not self.invalid_views

    def record_attempt(self, decision: ViewDecision, action: str = "invalid_capture") -> None:
        """Audit an invalid/unassigned acquisition without claiming view coverage.

        Useful when stability or face identity failed before a view capture was
        accepted. This preserves the RECHECK reason in the physical-part event.
        """
        if decision.verdict != "RECHECK":
            raise ValueError("Coverage-neutral attempts must have a RECHECK verdict")
        if self.final_decision is not None or self.state == SessionState.FAIL_LOCKED:
            raise RuntimeError("This event is locked; attempts cannot change its audit")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Audit attempt action is required")
        self.attempts.append({"timestamp": utc_now(), "action": action.strip(),
                              "decision": decision.to_dict()})

    def record(self, decision: ViewDecision) -> ViewDecision:
        if self.final_decision is not None or self.state == SessionState.FAIL_LOCKED:
            raise RuntimeError("This event is locked; explicitly reset or cancel before another capture")
        previous = self.decisions.get(decision.view)
        if previous is not None and previous.verdict != "RECHECK":
            raise ValueError("Only an invalid RECHECK capture may be replaced")
        snapshot = deepcopy(decision)
        self.decisions[decision.view] = snapshot
        self.attempts.append({"timestamp": utc_now(),
                              "action": "retake" if previous else "capture",
                              "decision": snapshot.to_dict()})
        if snapshot.verdict == "FAIL":
            self.state = SessionState.FAIL_LOCKED
        elif self.invalid_views:
            self.state = SessionState.RECHECK_REQUIRED
        elif self.missing_views:
            self.state = SessionState.WAITING_FOR_FLIP
        else:
            self.state = SessionState.READY_TO_FINALIZE
        return self.preview()

    def preview(self) -> ViewDecision:
        if self.final_decision is not None:
            return deepcopy(self.final_decision)
        metrics: dict[str, Any] = {
            "inspection_mode": self.inspection_mode,
            "required_views": list(self.required_views),
            "captured_views": list(self.decisions),
            "missing_views": self.missing_views,
            "invalid_views": self.invalid_views,
            "coverage": "configured views only; unobserved surfaces are not assessed",
            "view_verdicts": {v: d.verdict for v, d in self.decisions.items()},
            "model_versions": {v: d.model_version for v, d in self.decisions.items()},
        }
        if self.inspection_mode == "manual":
            metrics.update(decision_source="operator", inspection_scope="manual")
        candidates = []
        for view, decision in self.decisions.items():
            for candidate in decision.candidates:
                candidates.append({**deepcopy(candidate), "view": view})
        failed = [v for v, d in self.decisions.items() if d.verdict == "FAIL"]
        if failed:
            reasons = [f"{v}: {reason}" for v in failed for reason in self.decisions[v].reasons]
            if self.missing_views:
                reasons.append("Uncaptured views: " + ", ".join(self.missing_views))
            return ViewDecision("part", "FAIL", reasons or ["A view satisfied a rejection rule"], candidates, metrics)
        reasons = []
        if self.missing_views:
            reasons.append("Mandatory views missing: " + ", ".join(self.missing_views))
        for view in self.invalid_views:
            reasons.extend(f"{view}: {reason}" for reason in self.decisions[view].reasons)
            if not self.decisions[view].reasons:
                reasons.append(f"{view}: an acceptable retake is required")
        if reasons:
            return ViewDecision("part", "RECHECK", reasons, candidates, metrics)
        scopes = {str(d.metrics.get("inspection_scope", "visual")) for d in self.decisions.values()}
        metrics["inspection_scopes"] = sorted(scopes)
        if self.inspection_mode == "manual":
            reasons = ["All mandatory views accepted by the operator"]
        elif scopes == {"geometry_only"}:
            reasons = ["All mandatory views passed calibrated geometry checks; surface defects were not assessed"]
        else:
            reasons = ["All mandatory views passed their configured calibrated inspection checks"]
            if "geometry_only" in scopes:
                reasons.append("Some views were assessed for geometry only; review per-view coverage")
        return ViewDecision("part", "PASS", reasons, candidates, metrics)

    def finalize(self) -> ViewDecision:
        if self.final_decision is None:
            self.final_decision = self.preview()
            if self.inspection_mode == "manual":
                self.final_decision.metrics["display_decision"] = {
                    "PASS": "OK", "FAIL": "NG", "RECHECK": "View Recheck",
                }[self.final_decision.verdict]
            self.completed_at = utc_now()
            self.counted = True
            self.state = SessionState.RESULT_LOCKED
            self.attempts.append({"timestamp": self.completed_at, "action": "finalize",
                                  "verdict": self.final_decision.verdict})
        return deepcopy(self.final_decision)

    def cancel(self, reason: str = "Cancelled by operator") -> ViewDecision:
        if not reason.strip():
            raise ValueError("Cancellation requires a reason")
        if self.counted:
            raise RuntimeError("A counted event cannot be cancelled; explicitly reset for another event")
        if self.state == SessionState.CANCELLED:
            return deepcopy(self.final_decision)
        prior = self.preview()
        self.cancel_reason = reason.strip()
        self.completed_at = utc_now()
        self.state = SessionState.CANCELLED
        self.final_decision = ViewDecision("part", "RECHECK", [self.cancel_reason], prior.candidates,
                                           {**prior.metrics, "cancelled": True,
                                            "prior_verdict": prior.verdict})
        self.attempts.append({"timestamp": self.completed_at, "action": "cancel",
                              "reason": self.cancel_reason, "prior_decision": prior.to_dict()})
        return deepcopy(self.final_decision)

    def reset(self, reason: str = "Explicit operator reset") -> None:
        if not reason.strip():
            raise ValueError("Reset requires a reason")
        previous = self.to_dict()
        previous.pop("previous_events", None)
        self.reset_history.append({"timestamp": utc_now(), "reason": reason.strip(),
                                   "event": previous})
        self._begin()

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "part_id": self.part_id,
                "inspection_mode": self.inspection_mode,
                "created_at": self.created_at, "completed_at": self.completed_at,
                "required_views": list(self.required_views), "missing_views": self.missing_views,
                "state": self.state.value, "counted": self.counted,
                "cancel_reason": self.cancel_reason,
                "views": {v: d.to_dict() for v, d in self.decisions.items()},
                "attempts": deepcopy(self.attempts),
                "final_decision": self.final_decision.to_dict() if self.final_decision else None,
                "previous_events": deepcopy(self.reset_history)}
