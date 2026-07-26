from __future__ import annotations

from pathlib import Path

import src.nakayama_comprehensive_audit as audit


def test_exact_retarded_model_has_no_outside_cone_signal() -> None:
    result = audit.run_causality_audit(audit.AuditConfig())
    assert result["E_out"]["exact_retarded"] == 0.0
    assert result["E_out"]["causal_past_hold"] == 0.0


def test_arbitrary_interpolation_claim_has_counterexample() -> None:
    cfg = audit.AuditConfig()
    result = audit.run_causality_audit(cfg)
    assert result["universal_interpolation_claim_falsified"]
    assert result["E_out"]["centered_cubic"] > cfg.interpolation_falsification_threshold
    assert result["E_out"]["future_blend"] > cfg.interpolation_falsification_threshold


def test_unconstrained_euler_can_be_superluminal() -> None:
    result = audit.run_causality_audit(audit.AuditConfig())
    counterexample = result["integration_counterexample"]
    assert counterexample["euler_superluminal"]
    assert counterexample["rapidity_subluminal"]


def test_uniform_motion_lw_is_lorentz_covariant_within_tolerance() -> None:
    cfg = audit.AuditConfig(lorentz_trials=20)
    result = audit.run_lorentz_audit(cfg)
    assert result["analytic_tolerance_pass"]
    assert result["max_invariant_residual"] < 1.0e-10
    assert not result["strict_bitwise_covariance"]


def test_quantization_residual_decreases() -> None:
    cfg = audit.AuditConfig(lorentz_trials=20)
    result = audit.run_lorentz_audit(cfg)
    residuals = result["quantized_max_residuals"]
    assert residuals["1e-06"] < residuals["1e-02"]


def test_identity_and_classical_choi_are_cptp() -> None:
    static = audit.StaticAudit(
        source_root_exists=False,
        rust_file_count=0,
        rust_line_count=0,
        test_attribute_count=0,
        has_past_intersection=False,
        has_field_strength=False,
        has_tensor_boost=False,
        has_most_past_scheduler=False,
        has_multi_charge_superposition=False,
        explicitly_skips_self_force=False,
        has_information_field_symbol=False,
        has_icq_symbol=False,
        has_noether_symbol=False,
        has_quantum_process_symbol=False,
        source_digest_sha256="",
        build_statuses={},
    )
    result = audit.run_quantum_scope_audit(static)
    assert result["identity_channel_choi"]["is_cptp_within_tolerance"]
    assert result["classical_channel_embedding"]["is_cptp_within_tolerance"]


def test_static_audit_detects_self_force_skip(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "charge_set.rs").write_text(
        """
        fn field_strength_from_charges(charges: &[Charge], i: usize) {
            for (j, charge) in charges.iter().enumerate() {
                // ignore from self
                if i == j { continue; }
                let _ = charge.world_line.past_intersection();
            }
        }
        """,
        encoding="utf-8",
    )
    (source / "app.rs").write_text(
        "fs = lorentz * fs * lorentz.transposed();",
        encoding="utf-8",
    )
    result = audit.static_source_audit(tmp_path, {})
    assert result.explicitly_skips_self_force
    assert result.has_multi_charge_superposition
    assert result.has_tensor_boost
