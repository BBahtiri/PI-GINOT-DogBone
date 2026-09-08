#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the constitutive law, the plane-stress closure and the
hard boundary conditions.

Run with:  pytest tests/ -q

Every tolerance here is an assertion about *physics*, not about training
quality, so these must stay green on any change to the physics engine.
"""

import numpy as np
import pytest
import torch

import physics.neo_hookean as nh
from config import (DECODER_CONFIG, ENCODER_CONFIG, GEOMETRY_DEFAULT,
                    MATERIAL_CONFIG, get_fillet_geometry)
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone
from models.pi_ginot import PI_GINOT
from physics.neo_hookean import (deformation_gradient,
                                 first_piola_kirchhoff_stress,
                                 cauchy_stress, solve_F33_plane_stress)
from physics.uniaxial import (section_resultant_series,
                              section_resultant_small_strain,
                              section_resultant_uniform, uniaxial_P11,
                              uniaxial_lateral_stretch)

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
E = MATERIAL_CONFIG["E"]
NU = MATERIAL_CONFIG["nu"]

# Physically relevant range of the in-plane determinant.  Training operates
# near J2D = 1; the wider span guards the early-training excursions.
J_RANGE = torch.logspace(np.log10(0.05), np.log10(3.0), 41,
                         dtype=torch.float64).reshape(1, -1, 1)


@pytest.fixture(autouse=True)
def _check_every_solve():
    """Collect closure diagnostics on every solve inside the tests."""
    old = nh.F33_DIAG_EVERY
    nh.F33_DIAG_EVERY = 1
    yield
    nh.F33_DIAG_EVERY = old


def _exact_F33(J2D):
    """Closed-form plane-stress closure via Lambert W.

    From mu*F33^2 + lam*ln(J2D*F33) - mu = 0, substituting t = F33^2,

        F33 = sqrt( (lam/2mu) * W0( (2mu/lam) * exp(2c/lam) ) ),
        c   = mu - lam*ln(J2D)
    """
    lambertw = pytest.importorskip("scipy.special").lambertw
    J = np.asarray(J2D, dtype=float)
    c = MU - LAM * np.log(J)
    return np.sqrt((LAM / (2 * MU))
                   * np.real(lambertw((2 * MU / LAM) * np.exp(2 * c / LAM))))


def _old_newton_F33(J2D, n_iter=5):
    """The pre-0.1 closure: five unrolled Newton steps, graph attached.

    Kept verbatim as a reference for the value and first-derivative tests.
    """
    J2D_safe = torch.clamp(J2D, min=nh._DETF_MIN)
    F33 = torch.ones_like(J2D_safe)
    for _ in range(n_iter):
        J3D = torch.clamp(J2D_safe * F33, min=nh._DETF_MIN)
        logJ = torch.log(J3D)
        g = MU * F33 + (LAM * logJ - MU) / F33
        dg = MU + (LAM - LAM * logJ + MU) / (F33 ** 2 + 1e-12)
        F33 = torch.clamp(F33 - g / (dg + 1e-12), min=0.01)
    return F33


def _grads(du_dx, du_dy, dv_dx, dv_dy, scale=0.05, seed=0, n=512):
    g = torch.Generator().manual_seed(seed)
    return [(torch.rand(1, n, 1, generator=g, dtype=torch.float64) * 2 - 1) * scale
            for _ in range(4)]


# ----------------------------------------------------------------- closure --

def test_F33_matches_lambertw():
    """The Newton closure reproduces the exact Lambert-W root."""
    F33 = solve_F33_plane_stress(J_RANGE, MU, LAM)
    assert np.abs(F33.numpy() - _exact_F33(J_RANGE)).max() < 1e-14


def test_F33_converged_and_unclamped():
    """The solve reports a machine-precision residual and no clamping."""
    solve_F33_plane_stress(J_RANGE, MU, LAM)
    assert nh.LAST_F33_SOLVE["n_clamped"] == 0
    assert nh.LAST_F33_SOLVE["max_residual"] < 1e-13


def test_F33_first_derivative_matches_implicit_theorem():
    """dF33/dJ2D equals the analytic IFT value and the old unrolled path."""
    J = J_RANGE.clone().requires_grad_(True)
    g_new = torch.autograd.grad(solve_F33_plane_stress(J, MU, LAM).sum(), J)[0]

    F33 = torch.as_tensor(_exact_F33(J_RANGE))
    g_analytic = -(LAM / J_RANGE) / (2 * MU * F33 + LAM / F33)
    assert ((g_new - g_analytic).abs() / g_analytic.abs()).max() < 1e-12

    J_old = J_RANGE.clone().requires_grad_(True)
    g_old = torch.autograd.grad(_old_newton_F33(J_old).sum(), J_old)[0]
    assert ((g_new - g_old).abs() / g_old.abs()).max() < 1e-7


def test_F33_second_derivative():
    """d2F33/dJ2D2 must be right, not just the first derivative.

    This is the test that pins the closure's design.  equilibrium_residual
    differentiates P once w.r.t. x and the optimiser differentiates again
    w.r.t. the parameters, so the closure's *second* derivative is on the
    training path.  A custom autograd.Function whose backward is built from
    detached tensors -- the textbook implicit-function-theorem implementation
    -- reports this as zero and is off by ~14% here, silently.
    """
    def fd_reference(Jv, h_rel=1e-4):
        h = Jv * h_rel
        f = lambda x: float(_exact_F33(x))
        return (-f(Jv - 2 * h) + 16 * f(Jv - h) - 30 * f(Jv)
                + 16 * f(Jv + h) - f(Jv + 2 * h)) / (12 * h * h)

    for Jv in [0.6, 0.9, 1.0, 1.05, 1.3, 2.0]:
        J = torch.tensor(Jv, dtype=torch.float64, requires_grad=True)
        F33 = solve_F33_plane_stress(J, MU, LAM)
        d1 = torch.autograd.grad(F33, J, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, J)[0]
        ref = fd_reference(Jv)
        assert abs(d2.item() - ref) / abs(ref) < 1e-6, (
            f"J2D={Jv}: d2F33/dJ2D2 = {d2.item():.6e}, reference {ref:.6e}"
        )


def test_F33_gradcheck():
    """Autograd's own first- and second-order checks on the closure."""
    J = torch.linspace(0.3, 2.0, 12, dtype=torch.float64).requires_grad_(True)
    fn = lambda x: solve_F33_plane_stress(x, MU, LAM)
    assert torch.autograd.gradcheck(fn, (J,), eps=1e-6, atol=1e-8)
    assert torch.autograd.gradgradcheck(fn, (J,), eps=1e-6, atol=1e-6)


def test_P33_is_zero_plane_stress():
    """The closure actually enforces P33 = 0."""
    a = _grads(*[None] * 4, seed=7)
    F11, F12, F21, F22, J2D = deformation_gradient(*a)
    F33 = solve_F33_plane_stress(J2D, MU, LAM)
    P33 = MU * F33 + (LAM * torch.log(J2D * F33) - MU) / F33
    assert (P33.abs().max() / E).item() < 1e-12


def test_plane_strain_ignores_closure():
    """Plane strain must be F33 = 1 exactly, J = J2D."""
    a = _grads(*[None] * 4, seed=3)
    P = first_piola_kirchhoff_stress(*a, MU, LAM, "plane strain",
                                     return_J3D=True)
    assert torch.allclose(P[4], P[5])


# ------------------------------------------------------------ constitutive --

def test_rigid_translation_zero_stress():
    """A uniform displacement produces no stress in either formulation."""
    z = torch.zeros(1, 64, 1, dtype=torch.float64)
    for state in ("plane stress", "plane strain"):
        P11, P12, P21, P22, detF = first_piola_kirchhoff_stress(
            z, z, z, z, MU, LAM, state)
        assert torch.allclose(detF, torch.ones_like(detF))
        for comp in (P11, P12, P21, P22):
            assert comp.abs().max().item() / E < 1e-14


def test_stress_symmetry():
    """Angular momentum balance: P F^T must be symmetric.

    Only sigma_12 is computed in cauchy_stress, so nothing else in the code
    exercises the transposed index pairing.
    """
    for state in ("plane stress", "plane strain"):
        a = _grads(*[None] * 4, seed=11, scale=0.15)
        F11, F12, F21, F22, _ = deformation_gradient(*a)
        P11, P12, P21, P22, _ = first_piola_kirchhoff_stress(
            *a, MU, LAM, state)
        # (P F^T)_12 vs (P F^T)_21
        pf12 = P11 * F21 + P12 * F22
        pf21 = P21 * F11 + P22 * F12
        scale = max(pf12.abs().max().item(), 1e-30)
        assert (pf12 - pf21).abs().max().item() / scale < 1e-12


def test_cauchy_matches_definition():
    """sigma = (1/J3D) P F^T, with J3D from the same closure solve."""
    for state in ("plane stress", "plane strain"):
        a = _grads(*[None] * 4, seed=5)
        F11, F12, F21, F22, _ = deformation_gradient(*a)
        P11, P12, P21, P22, _, J3D = first_piola_kirchhoff_stress(
            *a, MU, LAM, state, return_J3D=True)
        S11, S22, S12, _ = cauchy_stress(*a, MU, LAM, state)
        assert torch.allclose(S11, (P11 * F11 + P12 * F12) / J3D)
        assert torch.allclose(S22, (P21 * F21 + P22 * F22) / J3D)
        assert torch.allclose(S12, (P11 * F21 + P12 * F22) / J3D)


def test_J3D_below_J2D_in_tension():
    """Under plane-stress tension F33 < 1, so J3D < J2D.

    This is why the barrier has to guard min(J2D, J3D).
    """
    e = torch.full((1, 32, 1), 0.04, dtype=torch.float64)
    z = torch.zeros_like(e)
    *_, J2D, J3D = first_piola_kirchhoff_stress(
        e, z, z, z, MU, LAM, "plane stress", return_J3D=True)
    assert (J3D < J2D).all()


def test_stress_state_required():
    """Omitting stress_state raises instead of silently picking plane strain."""
    z = torch.zeros(1, 4, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        first_piola_kirchhoff_stress(z, z, z, z, MU, LAM)
    with pytest.raises(ValueError):
        first_piola_kirchhoff_stress(z, z, z, z, MU, LAM, "plane-stress")


# ---------------------------------------------------------------- uniaxial --

@pytest.mark.parametrize("l1", [1.01, 1.037, 1.05, 1.10, 1.30])
def test_uniaxial_analytic(l1):
    """A homogeneous laterally-free stretch reproduces the closed form."""
    beta = float(uniaxial_lateral_stretch(l1, MU, LAM))
    du_dx = torch.full((1, 1, 1), l1 - 1.0, dtype=torch.float64)
    dv_dy = torch.full((1, 1, 1), beta - 1.0, dtype=torch.float64)
    z = torch.zeros_like(du_dx)

    P11, _, _, P22, _ = first_piola_kirchhoff_stress(
        du_dx, z, z, dv_dy, MU, LAM, "plane stress")

    assert abs(P22.item()) / E < 1e-12                    # laterally free
    ref = float(uniaxial_P11(l1, MU, LAM))
    assert abs(P11.item() - ref) / abs(ref) < 1e-10


def test_uniaxial_small_strain_limit():
    """P11 -> E*eps and beta -> 1 - nu*eps as the strain goes to zero."""
    eps = 1e-6
    assert abs(float(uniaxial_P11(1 + eps, MU, LAM)) / (E * eps) - 1.0) < 1e-4
    beta = float(uniaxial_lateral_stretch(1 + eps, MU, LAM))
    assert abs((1.0 - beta) / (NU * eps) - 1.0) < 1e-4


def test_F33_small_strain_sensitivity():
    """dF33/dJ2D at J2D = 1 is -nu/(1-nu), the plane-stress Poisson value."""
    J = torch.ones(1, dtype=torch.float64, requires_grad=True)
    d = torch.autograd.grad(solve_F33_plane_stress(J, MU, LAM).sum(), J)[0]
    assert abs(d.item() + NU / (1.0 - NU)) < 1e-12


# ------------------------------------------------------- resultant anchors --

def test_series_anchor_reduces_to_uniform_for_a_prismatic_bar():
    """With a near-constant section the varying-section solve must collapse
    onto the uniform-bar result."""
    fi = get_fillet_geometry(dict(L_total=54.0, W_grip=20.0, W_gauge=19.99,
                                  R_fillet=8.0, holes=[]))
    series = section_resultant_series(fi, 1.0, MU, LAM)
    uniform = section_resultant_uniform(fi, 1.0, MU, LAM)
    assert abs(series / uniform - 1.0) < 2e-3


@pytest.mark.parametrize("params", [
    dict(L_total=54.0, W_grip=20.0, W_gauge=10.0, R_fillet=12.0),
    dict(L_total=40.0, W_grip=26.0, W_gauge=6.0, R_fillet=20.0),
    dict(L_total=40.0, W_grip=20.0, W_gauge=6.0, R_fillet=8.0),
    dict(L_total=70.0, W_grip=26.0, W_gauge=6.0, R_fillet=20.0),
])
def test_legacy_anchor_undershoots(params):
    """The removed small-strain anchor sits below the varying-section truth.

    Pins the direction of the bias reported in commit 0.3: the stiff grip
    dominates the finite-strain correction, so the legacy target is too low,
    not too high.
    """
    fi = get_fillet_geometry({**params, "holes": []})
    series = section_resultant_series(fi, 1.0, MU, LAM)
    legacy = section_resultant_small_strain(fi, 1.0, E)
    assert legacy < series
    assert legacy / series < 0.96


# --------------------------------------------------------------- hard BCs --

@pytest.fixture(scope="module")
def model_and_geometry():
    torch.manual_seed(0)
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).eval()
    mesh = generate_dogbone(GEOMETRY_DEFAULT, rng=np.random.default_rng(0))
    coll = sample_collocation_points(mesh, n_interior=64, n_total_boundary=64,
                                     rng=np.random.default_rng(0))
    return model, mesh, coll


def test_hard_bc_exact(model_and_geometry):
    """u(0,y) = 0, u(L_half,y) = u_delta and v(x,0) = 0 hold exactly."""
    model, mesh, coll = model_and_geometry
    fi = mesh.fillet_info
    L_half, H_grip, H_gauge = fi["L_half"], fi["H_grip"], fi["H_gauge"]
    u_delta = 1.0

    bpc = torch.tensor(coll.boundary_pc, dtype=torch.float32).unsqueeze(0)
    x_m = torch.tensor([coll.x_max])
    y_m = torch.tensor([coll.y_max])
    u_d = torch.tensor([u_delta])
    with torch.no_grad():
        z = model.encode(bpc, x_m, y_m, sample_ids=torch.tensor([0]))

        ys = np.linspace(0.0, H_gauge, 17)
        left = torch.tensor(np.stack([np.zeros_like(ys), ys], -1),
                            dtype=torch.float32).unsqueeze(0)
        assert model.decode(left, z, u_d, x_m, y_m)[0, :, 0].abs().max() < 1e-6

        ys = np.linspace(0.0, H_grip, 17)
        grip = torch.tensor(np.stack([np.full_like(ys, L_half), ys], -1),
                            dtype=torch.float32).unsqueeze(0)
        u_grip = model.decode(grip, z, u_d, x_m, y_m)[0, :, 0]
        assert (u_grip - u_delta).abs().max() < 1e-5

        xs = np.linspace(0.0, L_half, 17)
        bot = torch.tensor(np.stack([xs, np.zeros_like(xs)], -1),
                           dtype=torch.float32).unsqueeze(0)
        assert model.decode(bot, z, u_d, x_m, y_m)[0, :, 1].abs().max() < 1e-6


def test_checkpoint_legacy_keys_are_stripped():
    """Checkpoints written before GeometryAuxHead was removed still load."""
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG)
    state = dict(model.state_dict())
    state["geom_aux_head.net.0.weight"] = torch.zeros(128, 64)
    state["geom_aux_head.net.0.bias"] = torch.zeros(128)
    clean, dropped = PI_GINOT.strip_legacy_keys(state)
    assert len(dropped) == 2
    model.load_state_dict(clean)


# ------------------------------------------------------------- dense eval --

def test_dense_eval_is_deterministic(tmp_path):
    """Rebuilding the frozen set from its seed reproduces it exactly.

    The artefact is committed, but it must also be regenerable: the numbers in
    the paper are only meaningful if the point set behind them can be rebuilt.
    """
    from eval.dense_eval import build_dense_eval_set, load_dense_eval_set

    params = [dict(GEOMETRY_DEFAULT),
              dict(L_total=44.0, W_grip=24.0, W_gauge=8.0, R_fillet=16.0,
                   holes=[])]
    a = build_dense_eval_set(params, n_interior=2048,
                             n_boundary_per_segment=64,
                             path=str(tmp_path / "a.npz"))
    b = build_dense_eval_set(params, n_interior=2048,
                             n_boundary_per_segment=64,
                             path=str(tmp_path / "b.npz"))
    for ga, gb in zip(a, b):
        assert np.array_equal(ga.interior, gb.interior)
        assert np.array_equal(ga.boundary_pc, gb.boundary_pc)

    loaded = load_dense_eval_set(str(tmp_path / "a.npz"))
    for ga, gl in zip(a, loaded):
        assert ga.params == gl.params
        assert np.array_equal(ga.interior, gl.interior)
        assert set(ga.boundary_pts) == set(gl.boundary_pts)


def test_dense_eval_points_are_inside_the_domain():
    """Every interior point of the frozen set satisfies the inside test."""
    from geometry.parametric_dogbone import _point_in_dogbone
    from eval.dense_eval import build_dense_eval_set

    g = build_dense_eval_set([dict(GEOMETRY_DEFAULT)], n_interior=4096,
                             n_boundary_per_segment=64, path=None)[0]
    x, y = g.interior[:, 0], g.interior[:, 1]
    assert _point_in_dogbone(x, y, g.fillet_info).all()


def test_val_bank_matches_the_trainer_builder():
    """val_bank_params reproduces the meshes the trainer validates on."""
    from config import TRAINING_CONFIG
    from geometry.banks import (VAL_BANK_SEED, build_geometry_bank,
                                val_bank_params)

    n = 4
    ranges = TRAINING_CONFIG["bank_geo_ranges"]
    bank = build_geometry_bank(n, ranges, holes_on=False, seed=VAL_BANK_SEED)
    assert [m.params for m, _ in bank] == val_bank_params(n, ranges)
