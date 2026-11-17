"""Explicit validation, technical-export, and deployment qualification gates."""

from dataclasses import asdict, dataclass
from typing import Any

from surveillance_ml.autotrain.config import AutoTrainConfig
from surveillance_ml.autotrain.dataset_audit import AuditResult


@dataclass(frozen=True)
class GateCheck:
    name: str
    status: str
    required: bool
    observed: float | bool | str | None
    threshold: float | bool | str | None
    reason: str


@dataclass(frozen=True)
class GateEvaluation:
    checks: tuple[GateCheck, ...]
    performance_thresholds_configured: bool
    performance_gates_passed: bool
    dataset_integrity_passed: bool
    candidate_qualified: bool
    technically_exportable: bool
    license_cleared: bool
    deployment_approved: bool
    qualification_status: str
    stopping_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric_check(name: str, value: Any, threshold: float | None) -> GateCheck:
    if threshold is None:
        return GateCheck(
            name,
            "not_configured",
            True,
            float(value) if isinstance(value, (int, float)) else None,
            None,
            "No numeric threshold is configured; this cannot count as a passed criterion.",
        )
    observed = float(value) if isinstance(value, (int, float)) else None
    if observed is None:
        return GateCheck(name, "failed", True, None, threshold, "Validation metric is missing.")
    passed = observed >= threshold
    return GateCheck(
        name,
        "passed" if passed else "failed",
        True,
        observed,
        threshold,
        "Metric meets the configured minimum."
        if passed
        else "Metric is below the configured minimum.",
    )


def evaluate_pre_export_gates(
    config: AutoTrainConfig,
    audit: AuditResult,
    metrics: dict[str, Any],
) -> GateEvaluation:
    """Evaluate measurable validation gates before allowing ONNX export."""
    thresholds = config.quality_gates
    checks = [
        _metric_check("validation_map50", metrics.get("mAP50"), thresholds.min_validation_map50),
        _metric_check(
            "firearm_ap50",
            metrics.get("per_class", {}).get("firearm", {}).get("average_precision_50"),
            thresholds.min_firearm_ap50,
        ),
        _metric_check(
            "knife_ap50",
            metrics.get("per_class", {}).get("knife", {}).get("average_precision_50"),
            thresholds.min_knife_ap50,
        ),
        _metric_check(
            "validation_precision",
            metrics.get("precision"),
            thresholds.min_validation_precision,
        ),
        _metric_check(
            "validation_recall",
            metrics.get("recall"),
            thresholds.min_validation_recall,
        ),
    ]
    fp_hour_threshold = thresholds.max_false_positives_per_hour
    if fp_hour_threshold is None:
        checks.append(
            GateCheck(
                "false_positives_per_hour",
                "not_configured",
                False,
                None,
                None,
                "Still-image evaluation cannot estimate false positives per hour.",
            )
        )
    else:
        checks.append(
            GateCheck(
                "false_positives_per_hour",
                "not_evaluated",
                True,
                None,
                fp_hour_threshold,
                "This gate requires representative timestamped video; "
                "no rate is inferred from images.",
            )
        )
    checks.append(
        GateCheck(
            "dataset_integrity",
            "passed" if audit.valid else "failed",
            True,
            audit.valid,
            True,
            "Dataset audit passed." if audit.valid else "Dataset audit has hard failures.",
        )
    )
    if thresholds.require_source_split_integrity:
        checks.append(
            GateCheck(
                "source_split_integrity",
                "passed" if audit.source_split_integrity else "failed",
                True,
                audit.source_split_integrity,
                True,
                "No source, exact-image, or perceptual-group leakage was found."
                if audit.source_split_integrity
                else "Source leakage or cross-split duplicate images were found.",
            )
        )
    performance_checks = [
        check
        for check in checks
        if check.name != "dataset_integrity" and check.name != "source_split_integrity"
    ]
    configured_checks = [check for check in performance_checks if check.status != "not_configured"]
    thresholds_configured = any(
        check.threshold is not None and check.name != "false_positives_per_hour"
        for check in configured_checks
    )
    performance_passed = thresholds_configured and all(
        check.status == "passed" for check in configured_checks
    )
    integrity_passed = audit.valid and (
        not thresholds.require_source_split_integrity or audit.source_split_integrity
    )
    reason = None
    if not thresholds_configured:
        reason = "No numeric validation-performance thresholds are configured."
    elif not performance_passed:
        reason = "One or more configured validation-performance gates did not pass."
    elif not integrity_passed:
        reason = "Dataset integrity gates did not pass."
    return GateEvaluation(
        checks=tuple(checks),
        performance_thresholds_configured=thresholds_configured,
        performance_gates_passed=performance_passed,
        dataset_integrity_passed=integrity_passed,
        candidate_qualified=False,
        technically_exportable=False,
        license_cleared=False,
        deployment_approved=False,
        qualification_status="pre_export",
        stopping_reason=reason,
    )


def finalize_qualification(
    config: AutoTrainConfig,
    pre_export: GateEvaluation,
    *,
    manifest_validated: bool | None,
    python_dotnet_parity: bool | None,
) -> GateEvaluation:
    """Add export, parity, license, and manual-approval outcomes to the gate result."""
    checks = list(pre_export.checks)
    if manifest_validated is not None or config.quality_gates.require_manifest_validation:
        checks.append(
            GateCheck(
                "manifest_validation",
                "passed" if manifest_validated else "failed",
                config.quality_gates.require_manifest_validation,
                manifest_validated,
                True,
                "Candidate manifest and model hash were validated."
                if manifest_validated
                else "Candidate manifest validation did not pass.",
            )
        )
    if python_dotnet_parity is not None or config.quality_gates.require_python_dotnet_parity:
        checks.append(
            GateCheck(
                "python_dotnet_parity",
                "passed" if python_dotnet_parity else "failed",
                config.quality_gates.require_python_dotnet_parity,
                python_dotnet_parity,
                True,
                "Actual non-empty validation detections agree across runtimes."
                if python_dotnet_parity
                else "Actual-detection Python/.NET parity was not verified.",
            )
        )
    license_cleared = not config.dataset.license_review_required or (
        config.dataset.license_review_status == "approved"
        and config.dataset.license.strip().lower() not in {"", "unknown", "unverified"}
    )
    checks.append(
        GateCheck(
            "license_review",
            "passed" if license_cleared else "blocked",
            config.dataset.license_review_required,
            config.dataset.license_review_status,
            "approved" if config.dataset.license_review_required else None,
            "License review status is approved."
            if license_cleared
            else "License uncertainty blocks deployment approval.",
        )
    )
    checks.append(
        GateCheck(
            "manual_deployment_approval",
            "not_granted",
            config.quality_gates.require_manual_deployment_approval,
            False,
            True if config.quality_gates.require_manual_deployment_approval else None,
            "AutoTrain never deploys automatically; explicit external approval is required.",
        )
    )
    manifest_passed = (
        manifest_validated is True
        if config.quality_gates.require_manifest_validation
        else manifest_validated is not False
    )
    parity_passed = (
        python_dotnet_parity is True
        if config.quality_gates.require_python_dotnet_parity
        else python_dotnet_parity is not False
    )
    technically_exportable = manifest_passed and parity_passed
    candidate_qualified = (
        pre_export.performance_gates_passed
        and pre_export.dataset_integrity_passed
        and technically_exportable
    )
    if not pre_export.performance_gates_passed:
        reason = pre_export.stopping_reason
        status = "not_qualified"
    elif not pre_export.dataset_integrity_passed:
        reason = "Dataset integrity gates did not pass."
        status = "not_qualified"
    elif not technically_exportable:
        reason = "Required export or runtime-compatibility gates did not pass."
        status = "not_qualified"
    elif not license_cleared:
        reason = "Candidate is technically qualified, but licensing is not cleared for deployment."
        status = "deployment_blocked"
    else:
        reason = "Candidate is qualified for manual review; no deployment approval was recorded."
        status = "manual_approval_required"
    return GateEvaluation(
        checks=tuple(checks),
        performance_thresholds_configured=pre_export.performance_thresholds_configured,
        performance_gates_passed=pre_export.performance_gates_passed,
        dataset_integrity_passed=pre_export.dataset_integrity_passed,
        candidate_qualified=candidate_qualified,
        technically_exportable=technically_exportable,
        license_cleared=license_cleared,
        deployment_approved=False,
        qualification_status=status,
        stopping_reason=reason,
    )
