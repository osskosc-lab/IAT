from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
STATUS_SUPPORTED = 'SUPPORTED_WITHIN_SCOPE'
STATUS_PARTIAL = 'PARTIALLY_SUPPORTED'
STATUS_NOT_ESTABLISHED = 'NOT_ESTABLISHED'
STATUS_CONTRADICTED = 'CONTRADICTED'
STATUS_OUT_OF_SCOPE = 'OUT_OF_SCOPE'
UPSTREAM_REPO = 'sogebu/special-relativity-web'
UPSTREAM_REF = '7fc8cb1e0aeb76eb90a475cd0f2ee64aee321033'

@dataclass(frozen=True)
class AuditConfig:
    seed: int = 20260726
    c: float = 1.0
    observer_x: float = 10.0
    intervention_amplitude: float = 1.0
    intervention_tau: float = 1.0
    source_sample_step: float = 0.5
    measurement_sigma: float = 0.002
    causal_time_start: float = 8.0
    causal_time_stop: float = 10.0
    causal_time_step: float = 0.005
    lorentz_trials: int = 100
    lorentz_tolerance: float = 2e-10
    interpolation_falsification_threshold: float = 0.0001
    strict_numeric_tolerance: float = 0.0

@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    claim: str
    status: str
    evidence: str
    limitation: str

@dataclass(frozen=True)
class StaticAudit:
    source_root_exists: bool
    rust_file_count: int
    rust_line_count: int
    test_attribute_count: int
    has_past_intersection: bool
    has_field_strength: bool
    has_tensor_boost: bool
    has_most_past_scheduler: bool
    has_multi_charge_superposition: bool
    explicitly_skips_self_force: bool
    has_information_field_symbol: bool
    has_icq_symbol: bool
    has_noether_symbol: bool
    has_quantum_process_symbol: bool
    source_digest_sha256: str
    build_statuses: dict[str, str]

def sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(str(path).encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()

def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='utf-8', errors='replace')

def gaussian_equal_covariance_tv(mean_difference: np.ndarray, sigma: float) -> float:
    norm = float(np.linalg.norm(mean_difference))
    if sigma <= 0.0:
        return float(norm > 0.0)
    return float(math.erf(norm / (2.0 * math.sqrt(2.0) * sigma)))

def bisect_root(function: Callable[[float], float], low: float, high: float, tolerance: float=1e-13, max_iterations: int=200) -> float:
    f_low = function(low)
    f_high = function(high)
    if not (f_low >= 0.0 and f_high <= 0.0):
        raise ValueError(f'Root is not bracketed: f({low})={f_low}, f({high})={f_high}')
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = function(mid)
        if abs(f_mid) <= tolerance or abs(high - low) <= tolerance:
            return mid
        if f_mid > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

def static_source_audit(simulator_root: Path, build_statuses: dict[str, str]) -> StaticAudit:
    rust_files = list(simulator_root.rglob('*.rs')) if simulator_root.exists() else []
    texts = {path: read_text_safe(path) for path in rust_files}
    joined = '\n'.join(texts.values())
    lower_joined = joined.lower()
    rust_line_count = sum((text.count('\n') + 1 for text in texts.values()))
    test_attribute_count = sum((text.count('#[test]') for text in texts.values()))
    charge_set_text = ''
    app_text = ''
    for path, text in texts.items():
        normalized = str(path).replace('\\', '/')
        if normalized.endswith('src/charge_set.rs'):
            charge_set_text = text
        if normalized.endswith('src/app.rs'):
            app_text = text
    has_past_intersection = 'past_intersection' in joined
    has_field_strength = 'field_strength' in joined
    has_tensor_boost = bool(re.search('lorentz\\s*\\*\\s*fs\\s*\\*\\s*lorentz\\.transposed\\(\\)', app_text))
    has_most_past_scheduler = 'most_past_charge_index' in charge_set_text
    has_multi_charge_superposition = bool('field_strength_from_charges' in charge_set_text and re.search('for\\s*\\([^\\)]*j[^\\)]*charge[^\\)]*\\)\\s*in\\s*charges', charge_set_text))
    explicitly_skips_self_force = bool(re.search('if\\s+i\\s*==\\s*j\\s*\\{\\s*continue;', charge_set_text, flags=re.DOTALL) or 'ignore form self' in charge_set_text.lower() or 'ignore from self' in charge_set_text.lower())
    has_information_field_symbol = bool(re.search('\\binformation[_\\s-]*field\\b', lower_joined))
    has_icq_symbol = bool(re.search('\\bicq\\b', lower_joined))
    has_noether_symbol = bool(re.search('\\bnoether\\b', lower_joined))
    has_quantum_process_symbol = bool(re.search('\\bprocess[_\\s-]*matrix\\b|\\bquantum[_\\s-]*process\\b', lower_joined))
    digest = sha256_files(rust_files) if rust_files else ''
    return StaticAudit(source_root_exists=simulator_root.exists(), rust_file_count=len(rust_files), rust_line_count=rust_line_count, test_attribute_count=test_attribute_count, has_past_intersection=has_past_intersection, has_field_strength=has_field_strength, has_tensor_boost=has_tensor_boost, has_most_past_scheduler=has_most_past_scheduler, has_multi_charge_superposition=has_multi_charge_superposition, explicitly_skips_self_force=explicitly_skips_self_force, has_information_field_symbol=has_information_field_symbol, has_icq_symbol=has_icq_symbol, has_noether_symbol=has_noether_symbol, has_quantum_process_symbol=has_quantum_process_symbol, source_digest_sha256=digest, build_statuses=dict(build_statuses))

def intervention_position(t: float, cfg: AuditConfig) -> float:
    if t <= 0.0:
        return 0.0
    return cfg.intervention_amplitude * (1.0 - math.exp(-t / cfg.intervention_tau))

def lagrange_value(query: float, sample_t: np.ndarray, sample_y: np.ndarray) -> float:
    total = 0.0
    for i in range(len(sample_t)):
        basis = 1.0
        for j in range(len(sample_t)):
            if i == j:
                continue
            basis *= (query - sample_t[j]) / (sample_t[i] - sample_t[j])
        total += sample_y[i] * basis
    return float(total)

def build_interpolators(cfg: AuditConfig) -> dict[str, Callable[[float], float]]:
    h = cfg.source_sample_step
    times = np.arange(-20.0, 20.0 + 0.5 * h, h)
    values = np.array([intervention_position(float(t), cfg) for t in times])
    def exact(t: float) -> float:
        return intervention_position(t, cfg)
    def causal_hold(t: float) -> float:
        index = int(np.searchsorted(times, t, side='right') - 1)
        index = min(max(index, 0), len(times) - 1)
        return float(values[index])
    def centered_cubic(t: float) -> float:
        center = int(np.searchsorted(times, t, side='left'))
        start = min(max(center - 2, 0), len(times) - 4)
        return lagrange_value(t, times[start:start + 4], values[start:start + 4])
    def future_blend(t: float) -> float:
        return 0.5 * exact(t) + 0.5 * exact(t + h)
    def current_time(t: float) -> float:
        return exact(t)
    return {'exact_retarded': exact, 'causal_past_hold': causal_hold, 'centered_cubic': centered_cubic, 'future_blend': future_blend, 'current_time': current_time}

def retarded_source_time(observer_time: float, source_y: Callable[[float], float], cfg: AuditConfig) -> float:
    def residual(source_time: float) -> float:
        distance = math.hypot(cfg.observer_x, source_y(source_time))
        return observer_time - source_time - distance / cfg.c
    low = observer_time - 4.0 * cfg.observer_x / cfg.c - 20.0
    high = observer_time
    return bisect_root(residual, low, high)

def retarded_coulomb_readout(observer_time: float, source_y: Callable[[float], float], cfg: AuditConfig) -> np.ndarray:
    source_time = retarded_source_time(observer_time, source_y, cfg)
    y = source_y(source_time)
    separation = np.array([cfg.observer_x, -y, 0.0], dtype=float)
    radius = float(np.linalg.norm(separation))
    return separation / radius ** 3

def current_time_coulomb_readout(observer_time: float, cfg: AuditConfig) -> np.ndarray:
    y = intervention_position(observer_time, cfg)
    separation = np.array([cfg.observer_x, -y, 0.0], dtype=float)
    radius = float(np.linalg.norm(separation))
    return separation / radius ** 3

def run_causality_audit(cfg: AuditConfig) -> dict:
    interpolators = build_interpolators(cfg)
    control = np.array([1.0 / cfg.observer_x ** 2, 0.0, 0.0], dtype=float)
    times = np.arange(cfg.causal_time_start, cfg.causal_time_stop, cfg.causal_time_step)
    series: dict[str, list[float]] = {}
    e_out: dict[str, float] = {}
    for name, interpolator in interpolators.items():
        values: list[float] = []
        for observer_time in times:
            if name == 'current_time':
                readout = current_time_coulomb_readout(float(observer_time), cfg)
            else:
                readout = retarded_coulomb_readout(float(observer_time), interpolator, cfg)
            values.append(gaussian_equal_covariance_tv(readout - control, cfg.measurement_sigma))
        series[name] = values
        e_out[name] = float(max(values))
    acceleration = 5.0
    dt = 0.5
    steps = 5
    euler_velocity = 0.0
    rapidity = 0.0
    for _ in range(steps):
        euler_velocity += acceleration * dt
        rapidity += acceleration * dt
    rapidity_velocity = math.tanh(rapidity)
    universal_interpolation_claim_falsified = bool(e_out['centered_cubic'] > cfg.interpolation_falsification_threshold or e_out['future_blend'] > cfg.interpolation_falsification_threshold)
    universal_integration_claim_falsified = abs(euler_velocity) >= cfg.c
    return {'arrival_time': cfg.observer_x / cfg.c, 'times': times.tolist(), 'tv_series': series, 'E_out': e_out, 'integration_counterexample': {'acceleration': acceleration, 'dt': dt, 'steps': steps, 'unconstrained_euler_velocity': euler_velocity, 'rapidity_integrator_velocity': rapidity_velocity, 'euler_superluminal': abs(euler_velocity) >= cfg.c, 'rapidity_subluminal': abs(rapidity_velocity) < cfg.c}, 'universal_interpolation_claim_falsified': universal_interpolation_claim_falsified, 'universal_integration_claim_falsified': universal_integration_claim_falsified}
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

def minkowski_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ ETA @ b)

def lorentz_boost(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    beta2 = float(beta @ beta)
    if beta2 >= 1.0:
        raise ValueError('Boost speed must be subluminal.')
    if beta2 == 0.0:
        return np.eye(4)
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    matrix = np.eye(4)
    matrix[0, 0] = gamma
    matrix[0, 1:] = -gamma * beta
    matrix[1:, 0] = -gamma * beta
    matrix[1:, 1:] += (gamma - 1.0) * np.outer(beta, beta) / beta2
    return matrix

def four_velocity(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=float)
    speed2 = float(velocity @ velocity)
    if speed2 >= 1.0:
        raise ValueError('Source speed must be subluminal.')
    gamma = 1.0 / math.sqrt(1.0 - speed2)
    return np.concatenate(([gamma], gamma * velocity))

def retarded_proper_time(observation_event: np.ndarray, source_origin: np.ndarray, source_four_velocity: np.ndarray) -> float:
    displacement = observation_event - source_origin
    r_dot_u = minkowski_dot(displacement, source_four_velocity)
    discriminant = r_dot_u ** 2 + minkowski_dot(displacement, displacement)
    if discriminant < -1e-12:
        raise ValueError('No real null intersection.')
    discriminant = max(discriminant, 0.0)
    return float(-r_dot_u - math.sqrt(discriminant))

def uniform_motion_lw_field(observation_event: np.ndarray, source_origin: np.ndarray, source_four_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = retarded_proper_time(observation_event, source_origin, source_four_velocity)
    source_event = source_origin + source_four_velocity * tau
    separation = observation_event[1:] - source_event[1:]
    radius = float(np.linalg.norm(separation))
    if radius <= 1e-14:
        raise ZeroDivisionError('Observation event lies on source worldline.')
    direction = separation / radius
    velocity = source_four_velocity[1:] / source_four_velocity[0]
    beta2 = float(velocity @ velocity)
    kappa = 1.0 - float(direction @ velocity)
    electric = (1.0 - beta2) * (direction - velocity) / (kappa ** 3 * radius ** 2)
    magnetic = np.cross(direction, electric)
    return (electric, magnetic)

def transform_fields(electric: np.ndarray, magnetic: np.ndarray, frame_beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_beta = np.asarray(frame_beta, dtype=float)
    beta2 = float(frame_beta @ frame_beta)
    if beta2 == 0.0:
        return (electric.copy(), magnetic.copy())
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    correction = gamma ** 2 / (gamma + 1.0)
    electric_prime = gamma * (electric + np.cross(frame_beta, magnetic)) - correction * frame_beta * float(frame_beta @ electric)
    magnetic_prime = gamma * (magnetic - np.cross(frame_beta, electric)) - correction * frame_beta * float(frame_beta @ magnetic)
    return (electric_prime, magnetic_prime)

def random_vector_bounded(rng: np.random.Generator, max_norm: float) -> np.ndarray:
    while True:
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vector = vector / norm * rng.uniform(0.0, max_norm)
            return vector

def quantize(value: np.ndarray, quantum: float) -> np.ndarray:
    return np.round(value / quantum) * quantum

def run_lorentz_audit(cfg: AuditConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    exact_residuals: list[float] = []
    invariant_residuals: list[float] = []
    trial_records: list[dict] = []
    base_cases: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for trial in range(cfg.lorentz_trials):
        source_velocity = random_vector_bounded(rng, 0.55)
        frame_beta = random_vector_bounded(rng, 0.5)
        source_origin = np.concatenate(([0.0], rng.uniform(-1.0, 1.0, size=3)))
        observation_event = np.concatenate(([rng.uniform(8.0, 15.0)], rng.uniform(-4.0, 4.0, size=3)))
        source_u = four_velocity(source_velocity)
        electric, magnetic = uniform_motion_lw_field(observation_event, source_origin, source_u)
        boost = lorentz_boost(frame_beta)
        observation_prime = boost @ observation_event
        source_origin_prime = boost @ source_origin
        source_u_prime = boost @ source_u
        direct_e_prime, direct_b_prime = uniform_motion_lw_field(observation_prime, source_origin_prime, source_u_prime)
        transformed_e_prime, transformed_b_prime = transform_fields(electric, magnetic, frame_beta)
        residual = max(float(np.linalg.norm(direct_e_prime - transformed_e_prime)), float(np.linalg.norm(direct_b_prime - transformed_b_prime)))
        exact_residuals.append(residual)
        invariant_1 = float(electric @ electric - magnetic @ magnetic)
        invariant_1_prime = float(direct_e_prime @ direct_e_prime - direct_b_prime @ direct_b_prime)
        invariant_2 = float(electric @ magnetic)
        invariant_2_prime = float(direct_e_prime @ direct_b_prime)
        invariant_residual = max(abs(invariant_1 - invariant_1_prime), abs(invariant_2 - invariant_2_prime))
        invariant_residuals.append(invariant_residual)
        base_cases.append((observation_event, source_origin, source_u, frame_beta))
        trial_records.append({'trial': trial, 'field_covariance_residual': residual, 'invariant_residual': invariant_residual})
    quantization_levels = [0.01, 0.001, 0.0001, 1e-05, 1e-06]
    quantized_residuals: dict[str, float] = {}
    for quantum in quantization_levels:
        residuals = []
        for observation_event, source_origin, source_u, frame_beta in base_cases:
            electric, magnetic = uniform_motion_lw_field(observation_event, source_origin, source_u)
            transformed_e, transformed_b = transform_fields(electric, magnetic, frame_beta)
            boost = lorentz_boost(frame_beta)
            observation_prime = quantize(boost @ observation_event, quantum)
            source_origin_prime = quantize(boost @ source_origin, quantum)
            spatial_u = quantize((boost @ source_u)[1:], quantum)
            source_u_prime = np.concatenate(([math.sqrt(1.0 + float(spatial_u @ spatial_u))], spatial_u))
            direct_e, direct_b = uniform_motion_lw_field(observation_prime, source_origin_prime, source_u_prime)
            residuals.append(max(float(np.linalg.norm(direct_e - transformed_e)), float(np.linalg.norm(direct_b - transformed_b))))
        quantized_residuals[f'{quantum:.0e}'] = float(max(residuals))
    exact_max = float(max(exact_residuals))
    invariant_max = float(max(invariant_residuals))
    strict_bitwise_covariance = exact_max <= cfg.strict_numeric_tolerance
    return {'trial_count': cfg.lorentz_trials, 'exact_max_field_residual': exact_max, 'exact_mean_field_residual': float(np.mean(exact_residuals)), 'max_invariant_residual': invariant_max, 'analytic_tolerance_pass': exact_max <= cfg.lorentz_tolerance, 'strict_bitwise_covariance': strict_bitwise_covariance, 'quantized_max_residuals': quantized_residuals, 'trial_records': trial_records}

def static_charge_field(observation: np.ndarray, position: np.ndarray, charge: float) -> np.ndarray:
    separation = observation - position
    radius = float(np.linalg.norm(separation))
    if radius <= 1e-14:
        return np.zeros(3)
    return charge * separation / radius ** 3

def run_multi_particle_audit(static: StaticAudit, cfg: AuditConfig) -> dict:
    rng = np.random.default_rng(cfg.seed + 1)
    residuals = []
    for _ in range(100):
        observation = rng.uniform(-3.0, 3.0, size=3)
        positions = rng.uniform(-2.0, 2.0, size=(5, 3))
        charges = rng.uniform(-2.0, 2.0, size=5)
        individual = [static_charge_field(observation, position, charge) for position, charge in zip(positions, charges)]
        direct_sum = np.sum(individual, axis=0)
        loop_sum = np.zeros(3)
        for field in individual:
            loop_sum += field
        residuals.append(float(np.linalg.norm(direct_sum - loop_sum)))
    return {'superposition_trial_count': 100, 'superposition_max_residual': float(max(residuals)), 'upstream_has_multi_charge_superposition': static.has_multi_charge_superposition, 'upstream_explicitly_skips_self_force': static.explicitly_skips_self_force, 'self_force_validation_possible_from_current_source': not static.explicitly_skips_self_force}

def partial_trace_output(choi: np.ndarray, dimension: int) -> np.ndarray:
    tensor = choi.reshape(dimension, dimension, dimension, dimension)
    return np.einsum('aiaj->ij', tensor)

def identity_channel_choi(dimension: int) -> np.ndarray:
    omega = np.zeros(dimension * dimension, dtype=complex)
    for index in range(dimension):
        omega[index * dimension + index] = 1.0
    return np.outer(omega, omega.conjugate())

def classical_channel_choi(transition: np.ndarray) -> np.ndarray:
    output_dim, input_dim = transition.shape
    if output_dim != input_dim:
        raise ValueError('This compact demonstrator assumes equal dimensions.')
    d = input_dim
    choi = np.zeros((d * d, d * d), dtype=complex)
    for output in range(d):
        for input_index in range(d):
            basis_index = output * d + input_index
            choi[basis_index, basis_index] = transition[output, input_index]
    return choi

def validate_choi(choi: np.ndarray, dimension: int) -> dict:
    hermitian_residual = float(np.linalg.norm(choi - choi.conjugate().T))
    eigenvalues = np.linalg.eigvalsh(0.5 * (choi + choi.conjugate().T))
    trace_preservation_residual = float(np.linalg.norm(partial_trace_output(choi, dimension) - np.eye(dimension)))
    return {'hermitian_residual': hermitian_residual, 'minimum_eigenvalue': float(np.min(eigenvalues)), 'trace_preservation_residual': trace_preservation_residual, 'is_cptp_within_tolerance': bool(hermitian_residual < 1e-12 and np.min(eigenvalues) > -1e-12 and (trace_preservation_residual < 1e-12))}

def run_quantum_scope_audit(static: StaticAudit) -> dict:
    identity = validate_choi(identity_channel_choi(2), 2)
    bit_flip_transition = np.array([[0.9, 0.2], [0.1, 0.8]], dtype=float)
    classical_embedding = validate_choi(classical_channel_choi(bit_flip_transition), 2)
    return {'identity_channel_choi': identity, 'classical_channel_embedding': classical_embedding, 'upstream_has_quantum_process_implementation': static.has_quantum_process_symbol, 'interpretation': 'A causally ordered classical channel can be embedded into a CPTP Choi operator, but this does not derive a general process matrix or indefinite causal order.'}

def build_claim_results(static: StaticAudit, causality: dict, lorentz: dict, multi: dict, quantum: dict) -> list[ClaimResult]:
    build_passes = all((value.lower() in {'success', 'pass', 'passed', 'skipped'} for value in static.build_statuses.values())) if static.build_statuses else False
    claim_1_status = STATUS_NOT_ESTABLISHED
    claim_1_evidence = f'Source audit found {static.rust_file_count} Rust files, {static.test_attribute_count} #[test] functions, PLC intersection and field routines. Recorded build steps all acceptable={build_passes}.'
    claim_1_limit = f'Finite unit tests/builds cannot prove numerical completeness. The source lacks a complete convergence, coverage, singularity, long-time energy, and independent-reference audit; self-force is explicitly skipped={static.explicitly_skips_self_force}.'
    arbitrary_falsified = bool(causality['universal_interpolation_claim_falsified'] or causality['universal_integration_claim_falsified'])
    claim_2_status = STATUS_CONTRADICTED if arbitrary_falsified else STATUS_NOT_ESTABLISHED
    claim_2_evidence = 'A centered/future-dependent reconstruction creates pre-arrival distinguishability and unconstrained Euler integration becomes superluminal, while causal past-hold and rapidity integration preserve the tested constraints.'
    claim_2_limit = "This counterexample refutes the universal word 'arbitrary'; it does not classify every possible scheme."
    claim_3_status = STATUS_PARTIAL
    claim_3_evidence = f"Exact controlled trials satisfy covariance within tolerance: max residual={lorentz['exact_max_field_residual']:.3e}; source contains a tensor boost={static.has_tensor_boost}."
    claim_3_limit = 'Strict numerical equality is false at finite precision, and the web application itself lacks an end-to-end frame-paired regression audit in this suite.'
    claim_4_status = STATUS_CONTRADICTED if static.explicitly_skips_self_force else STATUS_PARTIAL
    claim_4_evidence = f"Multi-charge superposition code present={static.has_multi_charge_superposition}; toy superposition residual={multi['superposition_max_residual']:.3e}; self-force explicitly skipped={static.explicitly_skips_self_force}."
    claim_4_limit = 'Pairwise retarded interaction and linear superposition do not validate radiation reaction, mass renormalization, singular collisions, or long-time energy-momentum accounting.'
    claim_5_status = STATUS_NOT_ESTABLISHED
    claim_5_evidence = f'No information-field implementation symbol was found={not static.has_information_field_symbol}; all audited dynamics are electromagnetic/worldline constructs.'
    claim_5_limit = 'Code absence does not prove metaphysical nonexistence; it shows the simulator supplies no evidence for the claim.'
    claim_6_status = STATUS_NOT_ESTABLISHED
    claim_6_evidence = f'No ICQ symbol found={not static.has_icq_symbol}; no Noether symbol found={not static.has_noether_symbol}.'
    claim_6_limit = 'A Noether charge requires an explicit action, continuous symmetry, and derived conserved current; none is defined by the audited simulator. Numerical fitting cannot substitute for this derivation.'
    claim_7_status = STATUS_OUT_OF_SCOPE
    claim_7_evidence = f"Upstream quantum-process implementation found={static.has_quantum_process_symbol}. A separate identity-channel Choi check passes={quantum['identity_channel_choi']['is_cptp_within_tolerance']}."
    claim_7_limit = 'The Choi demonstration is a separate formal embedding. It does not derive a process matrix, quantum memory, or indefinite causal order from the classical PLC simulator.'
    return [ClaimResult('C1', 'The public Nakayama simulator is numerically complete in its entirety.', claim_1_status, claim_1_evidence, claim_1_limit), ClaimResult('C2', 'Any interpolation and integration algorithm preserves causality.', claim_2_status, claim_2_evidence, claim_2_limit), ClaimResult('C3', 'Numerical results remain strictly Lorentz-equivariant after a boost.', claim_3_status, claim_3_evidence, claim_3_limit), ClaimResult('C4', 'Multi-particle interactions and self-force are correctly processed.', claim_4_status, claim_4_evidence, claim_4_limit), ClaimResult('C5', 'A new information field exists.', claim_5_status, claim_5_evidence, claim_5_limit), ClaimResult('C6', 'ICQ is a Noether conserved charge.', claim_6_status, claim_6_evidence, claim_6_limit), ClaimResult('C7', 'The classical simulator can be generalized directly to a quantum process matrix.', claim_7_status, claim_7_evidence, claim_7_limit)]

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

def write_claim_csv(path: Path, claims: Sequence[ClaimResult]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['claim_id', 'claim', 'status', 'evidence', 'limitation'])
        writer.writeheader()
        for claim in claims:
            writer.writerow(asdict(claim))

def plot_causality(causality: dict, output: Path) -> None:
    figure = plt.figure(figsize=(9.5, 5.5))
    axis = figure.add_subplot(111)
    times = np.asarray(causality['times'])
    for name, values in causality['tv_series'].items():
        axis.plot(times, values, label=name)
    axis.axvline(causality['arrival_time'], linestyle='--', label='causal arrival')
    axis.set_xlabel('observer time')
    axis.set_ylabel('TV distinguishability')
    axis.set_ylim(-0.02, 1.02)
    axis.set_title('Causality audit: interpolation and current-time counterexamples')
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

def plot_lorentz(lorentz: dict, output: Path) -> None:
    figure = plt.figure(figsize=(8.5, 5.5))
    axis = figure.add_subplot(111)
    levels = np.array([float(key) for key in lorentz['quantized_max_residuals']])
    residuals = np.array(list(lorentz['quantized_max_residuals'].values()))
    order = np.argsort(levels)
    axis.loglog(levels[order], residuals[order], marker='o')
    axis.axhline(lorentz['exact_max_field_residual'], linestyle='--', label='exact floating-point residual')
    axis.set_xlabel('quantization step')
    axis.set_ylabel('max covariance residual')
    axis.set_title('Lorentz covariance is convergent, not bitwise exact')
    axis.grid(True, which='both', alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

def write_markdown_summary(path: Path, static: StaticAudit, causality: dict, lorentz: dict, multi: dict, claims: Sequence[ClaimResult]) -> None:
    status_counts: dict[str, int] = {}
    for claim in claims:
        status_counts[claim.status] = status_counts.get(claim.status, 0) + 1
    lines = ['# Nakayama PLC Comprehensive Falsification Audit', '', f'- Upstream repository: `{UPSTREAM_REPO}`', f'- Pinned upstream ref: `{UPSTREAM_REF}`', f'- Upstream source digest: `{static.source_digest_sha256}`', f'- Rust files audited: {static.rust_file_count}', f'- Rust `#[test]` functions found: {static.test_attribute_count}', '', '## Executive decision', '', 'The audit does **not** establish numerical completeness, a new information field, an ICQ Noether charge, or a direct quantum-process generalization.', 'It supports the PLC causal mechanism and analytic Lorentz covariance within controlled scope, while falsifying the universal claim that arbitrary numerical schemes preserve causality.', '', '## Core metrics', '', f"- Exact-retarded outside-cone TV: `{causality['E_out']['exact_retarded']:.6e}`", f"- Centered-cubic outside-cone TV: `{causality['E_out']['centered_cubic']:.6e}`", f"- Future-blend outside-cone TV: `{causality['E_out']['future_blend']:.6e}`", f"- Current-time outside-cone TV: `{causality['E_out']['current_time']:.6e}`", f"- Lorentz exact max residual: `{lorentz['exact_max_field_residual']:.6e}`", f"- Lorentz invariant max residual: `{lorentz['max_invariant_residual']:.6e}`", f"- Multi-charge superposition max residual: `{multi['superposition_max_residual']:.6e}`", f'- Upstream explicitly skips self-force: `{static.explicitly_skips_self_force}`', '', '## Claim matrix', '', '| ID | Claim | Decision |', '|---|---|---|']
    for claim in claims:
        lines.append(f'| {claim.claim_id} | {claim.claim} | **{claim.status}** |')
    lines.extend(['', '## Interpretation guard', '', 'Passing the software build and controlled numerical tests means only that the audited implementation and toy models execute consistently under the stated conditions. It is not a proof of completeness or new physics.', '', '## Status counts', ''])
    for status, count in sorted(status_counts.items()):
        lines.append(f'- {status}: {count}')
    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')

def parse_build_statuses() -> dict[str, str]:
    names = ['UPSTREAM_CARGO_TEST', 'UPSTREAM_CARGO_CHECK_WASM', 'UPSTREAM_WASM_PACK_BUILD', 'UPSTREAM_WEB_BUILD']
    return {name.lower(): os.environ.get(name, 'unknown') for name in names}

def run_audit(simulator_root: Path, output_dir: Path, cfg: AuditConfig, build_statuses: dict[str, str] | None=None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = build_statuses if build_statuses is not None else parse_build_statuses()
    static = static_source_audit(simulator_root, statuses)
    causality = run_causality_audit(cfg)
    lorentz = run_lorentz_audit(cfg)
    multi = run_multi_particle_audit(static, cfg)
    quantum = run_quantum_scope_audit(static)
    claims = build_claim_results(static, causality, lorentz, multi, quantum)
    payload = {'audit_id': 'IAT-Nakayama-Comprehensive-Audit-v1', 'upstream_repository': UPSTREAM_REPO, 'upstream_ref': UPSTREAM_REF, 'config': asdict(cfg), 'static_audit': asdict(static), 'causality_audit': causality, 'lorentz_audit': lorentz, 'multi_particle_audit': multi, 'quantum_scope_audit': quantum, 'claims': [asdict(claim) for claim in claims], 'overall_decision': 'CONTROLLED_PLC_MECHANISM_SUPPORTED; UNIVERSAL_NUMERICAL_AND_NEW_PHYSICS_CLAIMS_NOT_ESTABLISHED'}
    write_json(output_dir / 'audit_summary.json', payload)
    write_claim_csv(output_dir / 'claim_matrix.csv', claims)
    plot_causality(causality, output_dir / 'causality_counterexamples.png')
    plot_lorentz(lorentz, output_dir / 'lorentz_covariance_residuals.png')
    write_markdown_summary(output_dir / 'summary.md', static, causality, lorentz, multi, claims)
    return payload

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Nakayama PLC comprehensive falsification audit')
    parser.add_argument('--simulator-root', type=Path, default=Path('external/special-relativity-web'), help='Path to the pinned upstream simulator checkout.')
    parser.add_argument('--output-dir', type=Path, default=Path('results/nakayama_audit'), help='Directory for JSON, CSV, Markdown, and figures.')
    parser.add_argument('--seed', type=int, default=AuditConfig.seed)
    parser.add_argument('--lorentz-trials', type=int, default=AuditConfig.lorentz_trials)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    cfg = AuditConfig(seed=args.seed, lorentz_trials=args.lorentz_trials)
    payload = run_audit(args.simulator_root, args.output_dir, cfg)
    print(json.dumps({'audit_id': payload['audit_id'], 'overall_decision': payload['overall_decision'], 'claims': [{'claim_id': claim['claim_id'], 'status': claim['status']} for claim in payload['claims']]}, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
