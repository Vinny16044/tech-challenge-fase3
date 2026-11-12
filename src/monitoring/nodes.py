"""
Nós do LangGraph da Fase 4 (integração multimodal).

    validate_events -> consolidate -> evaluate_alerts -> apply_reviews
                    -> summarize -> build_report

Se nenhum evento for aceito, o fluxo vai direto de validate_events para
build_report (o relatório ainda registra as rejeições).

Responsável: Paola
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from typing import Any

from pydantic import ValidationError

from src.contracts import (
    MODULES,
    REVIEW_STATUS_BY_DECISION,
    REVIEW_STATUS_PENDING,
    SCHEMA_VERSION,
    MonitoringEventV1,
    ReviewDecision,
)
from src.monitoring.audit import AuditLogger
from src.monitoring.rules import DISCLAIMER, RULES_VERSION, evaluate_group
from src.monitoring.state import MonitoringState
from src.monitoring.summarizer import SummaryLLM, summarize

REPORT_VERSION = "1.0"
MAX_ERRORS_PER_EVENT = 10


def _utc_iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "loc": ".".join(str(part) for part in error["loc"]) or "(evento)",
            "msg": error["msg"],
        }
        for error in exc.errors()[:MAX_ERRORS_PER_EVENT]
    ]


def _maybe_str(payload: Any, key: str) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get(key), str):
        return payload[key]
    return None


class MonitoringGraphNodes:
    def __init__(
        self,
        audit_path: str | None = None,
        llm: SummaryLLM | None = None,
        model_name: str | None = None,
    ) -> None:
        self.audit_path = audit_path
        self.llm = llm
        self.model_name = model_name

    def _audit(self, state: MonitoringState) -> AuditLogger:
        return AuditLogger(self.audit_path, run_id=state.get("run_id", ""))

    # ------------------------------------------------------------------ #
    # 1. Recebe, valida e deduplica os eventos
    # ------------------------------------------------------------------ #
    def validate_events(self, state: MonitoringState) -> dict[str, Any]:
        audit = self._audit(state)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        received = 0

        def reject(origin: str, reason: str, payload: Any = None, **details: Any) -> None:
            item = {
                "origin": origin,
                "reason": reason,
                "event_id": _maybe_str(payload, "event_id"),
                "patient_id": _maybe_str(payload, "patient_id"),
                "module": _maybe_str(payload, "module"),
                **details,
            }
            rejected.append(item)
            audit.log("evento_rejeitado", **item)

        for raw in state.get("raw_inputs", []):
            origin = raw["origin"]
            content = raw["content"]

            if isinstance(content, (str, bytes)):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    received += 1
                    reject(origin, "invalid_json", error=str(exc))
                    continue

            if isinstance(content, dict):
                items = [(origin, content)]
            elif isinstance(content, list):
                items = [(f"{origin}#{index}", item) for index, item in enumerate(content)]
            else:
                received += 1
                reject(origin, "invalid_payload_type", error=type(content).__name__)
                continue

            for item_origin, payload in items:
                received += 1
                if not isinstance(payload, dict):
                    reject(item_origin, "invalid_payload_type", error=type(payload).__name__)
                    continue

                try:
                    event = MonitoringEventV1.model_validate(payload)
                except ValidationError as exc:
                    reject(
                        item_origin,
                        "schema_validation_error",
                        payload,
                        errors=_validation_errors(exc),
                    )
                    continue

                content_hash = event.content_hash()
                previous = seen.get(event.event_id)
                if previous is not None:
                    if previous["content_hash"] == content_hash:
                        duplicate = {
                            "origin": item_origin,
                            "event_id": event.event_id,
                            "first_origin": previous["origin"],
                            "reason": "duplicate_resend_ignored",
                        }
                        duplicates.append(duplicate)
                        audit.log("evento_duplicado_ignorado", **duplicate)
                    else:
                        reject(
                            item_origin,
                            "event_id_conflict",
                            payload,
                            error=(
                                "event_id já recebido com conteúdo diferente; "
                                "o primeiro recebido foi mantido"
                            ),
                            first_origin=previous["origin"],
                        )
                    continue

                record = {"event": event, "origin": item_origin, "content_hash": content_hash}
                seen[event.event_id] = record
                accepted.append(record)
                audit.log(
                    "evento_aceito",
                    origin=item_origin,
                    event_id=event.event_id,
                    patient_id=event.patient_id,
                    module=event.module,
                    event_type=event.event_type,
                    timestamp=event.timestamp.isoformat(),
                    anomaly_detected=event.anomaly_detected,
                    content_hash=content_hash,
                )

        return {
            "accepted_events": accepted,
            "rejected_events": rejected,
            "duplicate_events": duplicates,
            "events_received": received,
            "status": "EVENTS_VALIDATED",
        }

    @staticmethod
    def route_after_validation(state: MonitoringState) -> str:
        return "consolidate" if state.get("accepted_events") else "build_report"

    # ------------------------------------------------------------------ #
    # 2. Consolida por paciente e janela de tempo
    # ------------------------------------------------------------------ #
    def consolidate(self, state: MonitoringState) -> dict[str, Any]:
        window_seconds = int(state["window_minutes"]) * 60
        buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}

        for record in state.get("accepted_events", []):
            event: MonitoringEventV1 = record["event"]
            epoch = int(event.timestamp.timestamp())
            bucket = epoch - (epoch % window_seconds)
            buckets.setdefault((event.patient_id, bucket), []).append(
                {
                    "event_id": event.event_id,
                    "module": event.module,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "timestamp_utc": _utc_iso(event.timestamp),
                    "finding": event.finding,
                    "anomaly_detected": event.anomaly_detected,
                    "priority": event.priority,
                    "evidence": [
                        item.model_dump(mode="json", exclude_none=True)
                        for item in event.evidence
                    ],
                    "source": event.source.model_dump(mode="json", exclude_none=True),
                    "origin": record["origin"],
                }
            )

        groups: list[dict[str, Any]] = []
        for (patient_id, bucket), events in sorted(buckets.items()):
            start = dt.datetime.fromtimestamp(bucket, dt.timezone.utc)
            end = start + dt.timedelta(seconds=window_seconds)
            events.sort(
                key=lambda e: (e["timestamp_utc"], MODULES.index(e["module"]), e["event_id"])
            )
            received = [m for m in MODULES if any(e["module"] == m for e in events)]
            groups.append(
                {
                    "group_id": f"{patient_id}@{_utc_iso(start)}",
                    "patient_id": patient_id,
                    "window_start": _utc_iso(start),
                    "window_end": _utc_iso(end),
                    "status": "consolidated",
                    "sources_received": received,
                    "sources_missing": [m for m in MODULES if m not in received],
                    "events": events,
                    "alert": None,
                    "summary": None,
                }
            )

        return {"groups": groups, "status": "EVENTS_CONSOLIDATED"}

    # ------------------------------------------------------------------ #
    # 3. Regras de alerta
    # ------------------------------------------------------------------ #
    def evaluate_alerts(self, state: MonitoringState) -> dict[str, Any]:
        audit = self._audit(state)
        groups = copy.deepcopy(state.get("groups", []))

        for group in groups:
            alert = evaluate_group(group)
            group["alert"] = alert
            if alert is None:
                group["status"] = "no_alert"
                audit.log(
                    "sem_alerta",
                    group_id=group["group_id"],
                    patient_id=group["patient_id"],
                    window_start=group["window_start"],
                    event_ids=[e["event_id"] for e in group["events"]],
                    sources_received=group["sources_received"],
                )
                continue

            group["status"] = "alert_pending_review"
            audit.log(
                "alerta_gerado",
                alert_id=alert["alert_id"],
                group_id=group["group_id"],
                patient_id=alert["patient_id"],
                window_start=alert["window_start"],
                window_end=alert["window_end"],
                priority=alert["priority"],
                rules_version=alert["rules_version"],
                rules=[r["rule_id"] for r in alert["priority_reasons"]],
                event_ids=alert["event_ids"],
                evidence=[
                    {"event_id": item["event_id"], "uri": item["uri"]}
                    for item in alert["evidence"]
                ],
                sources_received=group["sources_received"],
                sources_missing=group["sources_missing"],
                review_status=alert["review"]["status"],
            )

        return {"groups": groups, "status": "ALERTS_EVALUATED"}

    # ------------------------------------------------------------------ #
    # 4. Revisão humana (decisões informadas)
    # ------------------------------------------------------------------ #
    def apply_reviews(self, state: MonitoringState) -> dict[str, Any]:
        audit = self._audit(state)
        groups = copy.deepcopy(state.get("groups", []))
        results: list[dict[str, Any]] = []
        alerts = {g["alert"]["alert_id"]: g for g in groups if g.get("alert")}

        for raw in state.get("review_decisions") or []:
            alert_id = _maybe_str(raw, "alert_id")
            try:
                decision = ReviewDecision.model_validate(raw)
            except ValidationError as exc:
                item = {
                    "alert_id": alert_id,
                    "outcome": "rejected",
                    "reason": "invalid_review_decision",
                    "errors": _validation_errors(exc),
                }
                results.append(item)
                audit.log("revisao_rejeitada", **item)
                continue

            group = alerts.get(decision.alert_id)
            reason = None
            if group is None:
                reason = "alert_not_found"
            elif group["alert"]["review"]["status"] != REVIEW_STATUS_PENDING:
                reason = "alert_already_reviewed"
            if reason:
                item = {
                    "alert_id": decision.alert_id,
                    "reviewer_id": decision.reviewer_id,
                    "outcome": "rejected",
                    "reason": reason,
                }
                results.append(item)
                audit.log("revisao_rejeitada", **item)
                continue

            review = {
                "status": REVIEW_STATUS_BY_DECISION[decision.decision],
                "reviewer_id": decision.reviewer_id,
                "reviewer_name": decision.reviewer_name,
                "decision": decision.decision,
                "decided_at": decision.decided_at.isoformat(),
                "notes": decision.notes,
            }
            group["alert"]["review"] = review
            group["status"] = "alert_reviewed"
            item = {"alert_id": decision.alert_id, "outcome": "recorded", **review}
            results.append(item)
            audit.log(
                "revisao_registrada",
                patient_id=group["patient_id"],
                group_id=group["group_id"],
                event_ids=group["alert"]["event_ids"],
                **item,
            )

        return {"groups": groups, "review_results": results, "status": "REVIEWS_APPLIED"}

    # ------------------------------------------------------------------ #
    # 5. Resumo (determinístico; LLM opcional)
    # ------------------------------------------------------------------ #
    def summarize(self, state: MonitoringState) -> dict[str, Any]:
        audit = self._audit(state)
        groups = copy.deepcopy(state.get("groups", []))
        for group in groups:
            group["summary"] = summarize(group, llm=self.llm, model_name=self.model_name)
            audit.log(
                "resumo_gerado",
                group_id=group["group_id"],
                alert_id=(group["alert"] or {}).get("alert_id"),
                method=group["summary"]["method"],
                model=group["summary"]["model"],
                fallback_reason=group["summary"]["fallback_reason"],
            )
        return {"groups": groups, "status": "SUMMARIZED"}

    # ------------------------------------------------------------------ #
    # 6. Relatório estruturado
    # ------------------------------------------------------------------ #
    def build_report(self, state: MonitoringState) -> dict[str, Any]:
        audit = self._audit(state)
        groups = state.get("groups", [])
        accepted = state.get("accepted_events", [])
        rejected = state.get("rejected_events", [])
        duplicates = state.get("duplicate_events", [])

        received_modules = [
            m for m in MODULES if any(r["event"].module == m for r in accepted)
        ]
        alerts = [
            {
                "alert_id": g["alert"]["alert_id"],
                "patient_id": g["patient_id"],
                "window_start": g["window_start"],
                "priority": g["alert"]["priority"],
                "review_status": g["alert"]["review"]["status"],
            }
            for g in groups
            if g.get("alert")
        ]

        report = {
            "report_version": REPORT_VERSION,
            "contract_version": SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "run_id": state.get("run_id"),
            "generated_at": state.get("generated_at"),
            "config": {
                "window_minutes": state.get("window_minutes"),
                "summary_mode": "llm" if self.llm else "offline",
                "model": self.model_name if self.llm else None,
            },
            "disclaimer": DISCLAIMER,
            "intake": {
                "inputs_received": len(state.get("raw_inputs", [])),
                "events_received": state.get("events_received", 0),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "duplicates": len(duplicates),
                "accepted_events": [
                    {"event_id": r["event"].event_id, "module": r["event"].module, "origin": r["origin"]}
                    for r in accepted
                ],
                "rejected_events": rejected,
                "duplicate_events": duplicates,
            },
            "sources_received": received_modules,
            "sources_missing": [m for m in MODULES if m not in received_modules],
            "alerts": alerts,
            "groups": groups,
            "review_decisions": state.get("review_results", []),
            "audit_log": str(AuditLogger(self.audit_path).path),
        }

        audit.log(
            "execucao_concluida",
            events_received=report["intake"]["events_received"],
            accepted=len(accepted),
            rejected=len(rejected),
            duplicates=len(duplicates),
            group_ids=[g["group_id"] for g in groups],
            alert_ids=[a["alert_id"] for a in alerts],
        )

        return {"report": report, "status": "COMPLETED"}
