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


# --------------------------------------------------------------- manifest --

def test_manifest_hash_ignores_cosmetic_keys():
    """Two runs differing only in print frequency share a config hash."""
    import config as config_module
    from training.manifest import collect_config, config_hash

    base = collect_config(config_module)
    tweaked = {**base,
               "TRAINING_CONFIG": {**base["TRAINING_CONFIG"],
                                   "print_every": 999, "plot_every": 7}}
    assert config_hash(tweaked) == config_hash(base)

    science = {**base,
               "TRAINING_CONFIG": {**base["TRAINING_CONFIG"],
                                   "w_equilibrium": 1.0}}
    assert config_hash(science) != config_hash(base)


def test_manifest_is_written_and_complete(tmp_path):
    """write_manifest records git state, environment and the config."""
    import json

    import config as config_module
    from training.manifest import MANIFEST_NAME, write_manifest

    m = write_manifest(str(tmp_path), config_module, extra={"device": "cpu"})
    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert on_disk["config_sha256"] == m["config_sha256"]
    assert on_disk["environment"]["torch"]
    assert on_disk["environment"]["numpy"]
    assert "MATERIAL_CONFIG" in on_disk["config"]
    assert set(on_disk["git"]) >= {"commit", "branch", "dirty"}
    assert on_disk["extra"]["device"] == "cpu"


def test_set_deterministic_leaves_the_model_usable():
    """Deterministic kernels must not break the encoder's scatter ops.

    pointnet2_utils has gather/scatter kernels without deterministic
    implementations; warn_only=True is what keeps them usable, and this test
    is the guard that the flag is doing its job.
    """
    import warnings

    from training.manifest import set_deterministic

    try:
        set_deterministic(0, warn_only=True)
        model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).eval()
        mesh = generate_dogbone(GEOMETRY_DEFAULT, rng=np.random.default_rng(0))
        coll = sample_collocation_points(mesh, n_interior=16,
                                         n_total_boundary=16,
                                         rng=np.random.default_rng(0))
        bpc = torch.tensor(coll.boundary_pc, dtype=torch.float32).unsqueeze(0)
        x_m = torch.tensor([coll.x_max])
        y_m = torch.tensor([coll.y_max])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                z = model.encode(bpc, x_m, y_m, sample_ids=torch.tensor([0]))
        assert torch.isfinite(z).all()
    finally:
        torch.use_deterministic_algorithms(False)


def test_epoch_history_keys_match_the_epoch_summary():
    """Every history channel must read a key _train_one_epoch actually returns.

    history["L_resultant"] read "L_resultant_log", which _train_one_epoch does
    not return, so the section-resultant curve was a flat zero in every run --
    on the one term the force-balance argument rests on.
    """
    import inspect
    import re

    from training import trainer as trainer_mod

    # accum and mins are expanded into the return dict with **, so every
    # string key literal in the method is a key the caller can read.
    src = inspect.getsource(trainer_mod.PI_GINOT_Trainer._train_one_epoch)
    returned = set(re.findall(r'"(\w+)":', src))

    fit_src = inspect.getsource(trainer_mod.PI_GINOT_Trainer.fit)
    read = set(re.findall(r'batch_loss(?:\.get)?[\[\(]"(\w+)"', fit_src))

    missing = sorted(k for k in read if k not in returned)
    assert not missing, f"fit() reads keys _train_one_epoch never returns: {missing}"


def test_every_history_channel_is_actually_recorded():
    """The reverse direction: a declared channel that is never appended.

    The key-matching test above catches fit() reading a key that does not
    exist.  It cannot catch the opposite failure, which is what happened to
    "L_energy": the energy form's own objective was accumulated per batch and
    printed in the epoch summary, but history never declared or appended it,
    so the one quantity being minimised was the one quantity not kept.  No key
    mismatch, no error -- just a channel silently missing from every run.

    Checked statically so this stays cheap: every channel declared in the
    history dict must be appended somewhere in fit().
    """
    import inspect
    import re

    from training import trainer as trainer_mod

    init_src = inspect.getsource(trainer_mod.PI_GINOT_Trainer.__init__)
    decl_block = init_src[init_src.index("self.history = {"):]
    decl_block = decl_block[:decl_block.index("}")]
    declared = set(re.findall(r'"(\w+)":', decl_block))

    fit_src = inspect.getsource(trainer_mod.PI_GINOT_Trainer.fit)
    appended = set(re.findall(r'self\.history\["(\w+)"\]\.append', fit_src))

    never = sorted(declared - appended)
    assert not never, (
        f"history channels declared but never appended in fit(): {never}")


# ----------------------------------------------------- verification level --

def test_operators_are_exact():
    """Level 0: the physics operators against closed-form fields.

    Rigid motions give zero stress, P is frame-indifferent, an affine field
    gives a constant and exactly correct stress, and on the real curved
    geometry a uniaxial field satisfies every boundary condition except the
    fillet arc -- which checks the normals, the tagging and the traction
    index pairing at once.
    """
    from verification import operators as ops

    for v in ops.check_zero_stress_states().values():
        assert v < 1e-15
    for v in ops.check_objectivity().values():
        assert v < 1e-13
    for k, v in ops.check_affine_patch_test(n=800).items():
        assert v < 1e-13, k

    bcs = ops.check_dogbone_boundary_conditions(n_per_segment=200)
    for seg in ("bottom", "right_grip", "gauge_top", "left_symmetry"):
        assert bcs[seg] < 1e-14, f"{seg} traction should vanish exactly"
    # A uniform stretch is not the dog-bone solution, so the fillet must not
    # be traction-free -- a zero here would mean the arc normals are dead.
    assert bcs["right_arc"] > 1e-3

    for k, v in ops.check_section_resultant(n_y=256).items():
        if not k.startswith("_"):
            assert v < 1e-13, k


@pytest.mark.parametrize("state", ["plane strain", "plane stress"])
def test_mms_divergence_matches_symbolic(state):
    """Level 1: Div P through the full second-order AD chain, against sympy.

    The reference is built from the closed-form Lambert-W solution of the
    plane-stress closure, independently of this repository's Newton solve, so
    a defect cannot cancel itself out.
    """
    from verification.mms import check_closure_identity, run_interior

    assert check_closure_identity(n=50) < 1e-13
    r = run_interior(state, n=400)
    assert r["div_rms_reference"] > 1.0, "reference divergence must be non-trivial"
    assert r["rel_L2"] < 1e-12
    assert r["rel_Linf"] < 1e-11


def test_mms_traction_matches_symbolic():
    """Level 1: P.N on every segment against the symbolic P*.N."""
    from verification.mms import run_boundary

    for seg, err in run_boundary("plane stress", n_per_segment=120).items():
        assert err < 1e-12, seg


def test_boundary_geometry_is_exact():
    """Level 0 (geometry): points, normals, outwardness and lengths.

    Normals are the load-bearing part: every traction term is P.N, so a
    flipped sign silently converts 'traction-free' into a constraint pulling
    the field somewhere else, and the training curve looks fine throughout.
    """
    from verification.geometry_checks import GEOMETRY_TOL, check_one_geometry

    cases = [
        dict(GEOMETRY_DEFAULT),
        dict(L_total=40.0, W_grip=26.0, W_gauge=6.0, R_fillet=20.0, holes=[]),
        dict(L_total=70.0, W_grip=16.5, W_gauge=16.0, R_fillet=8.0, holes=[]),
    ]
    for params in cases:
        r = check_one_geometry(params, n_per_segment=200)
        for key in ("on_boundary", "unit_normals", "normal_direction",
                    "length", "loop_gap"):
            assert r[key] < GEOMETRY_TOL, (key, params)
        assert r["outward_fail"] == 0, params
        assert r["inward_fail"] == 0, params


def test_measure_consistent_flag_fixes_the_boundary_split():
    """The opt-in flag restores arc-length proportionality at equal budget.

    Uniform-per-segment sampling leaves the encoder's point cloud up to 17x
    denser on the short symmetry face than on the long bottom face, and
    starves the collocation sampler's pool so it returns fewer points than
    asked for. The flag shares the same total out by length instead.
    """
    from verification.geometry_checks import check_one_geometry

    params = dict(L_total=40.0, W_grip=26.0, W_gauge=6.0, R_fillet=20.0,
                  holes=[])
    off = check_one_geometry(params, measure_consistent=False)
    on = check_one_geometry(params, measure_consistent=True)

    assert off["pc_density_ratio"] > 3.0
    assert on["pc_density_ratio"] < 1.6
    assert on["sampler_measure"] < 0.02
    assert on["boundary_shortfall"] < 1e-9
    # The total point budget must not change -- only its distribution.
    from geometry.parametric_dogbone import generate_dogbone
    tot = lambda mc: sum(len(s.points) for s in generate_dogbone(
        params, n_pts_per_segment=400, rng=np.random.default_rng(0),
        measure_consistent=mc).boundary_segments)
    assert tot(False) == tot(True)


# ---------------------------------------------------------- energy form --

def test_energy_gradient_is_the_stress():
    """dPsi/dF must equal P, for both stress states.

    Under plane stress this is the envelope theorem: Psi is the *reduced*
    energy with F33 eliminated by P33 = 0, so the dF33/dF path is multiplied
    by a residual the closure drives to ~1e-16 and contributes nothing. If
    this ever fails, the energy form and the stress form have diverged and
    every comparison between them is meaningless.
    """
    from physics.neo_hookean import strain_energy_density

    for state in ("plane stress", "plane strain"):
        a = _grads(*[None] * 4, seed=13, scale=0.12, n=256)
        a = [t.clone().requires_grad_(True) for t in a]
        W = strain_energy_density(*a, MU, LAM, state)
        grads = torch.autograd.grad(W.sum(), a)
        P = first_piola_kirchhoff_stress(*a, MU, LAM, state)[:4]
        scale = max(p.abs().max().item() for p in P)
        err = max((g - p).abs().max().item() for g, p in zip(grads, P))
        assert err / scale < 1e-13, state


@pytest.mark.parametrize("state", ["plane stress", "plane strain"])
def test_energy_vanishes_for_rigid_motion(state):
    """Psi = 0 at F = I and under a finite rigid rotation (objectivity)."""
    from physics.neo_hookean import strain_energy_density

    z = torch.zeros(1, 8, 1, dtype=torch.float64)
    assert strain_energy_density(z, z, z, z, MU, LAM, state).abs().max() == 0.0

    th = 0.4
    c, s = np.cos(th), np.sin(th)
    H = np.array([[c - 1.0, -s], [s, c - 1.0]])
    g = [torch.full((1, 4, 1), H[i, j], dtype=torch.float64)
         for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    assert strain_energy_density(*g, MU, LAM, state).abs().max() < 1e-14


def test_energy_matches_the_closed_form_uniaxial():
    """Psi under a homogeneous laterally-free stretch, against the 3D form."""
    from physics.neo_hookean import strain_energy_density

    l1 = 1.05
    beta = float(uniaxial_lateral_stretch(l1, MU, LAM))
    g = [torch.tensor([[[l1 - 1.0]]], dtype=torch.float64),
         torch.zeros(1, 1, 1, dtype=torch.float64),
         torch.zeros(1, 1, 1, dtype=torch.float64),
         torch.tensor([[[beta - 1.0]]], dtype=torch.float64)]
    W = strain_energy_density(*g, MU, LAM, "plane stress").item()

    J = l1 * beta * beta
    trC = l1 ** 2 + 2.0 * beta ** 2
    ref = 0.5 * MU * (trC - 3.0) - MU * np.log(J) + 0.5 * LAM * np.log(J) ** 2
    assert abs(W - ref) / abs(ref) < 1e-12


def test_stratified_quadrature_is_unbiased_and_moves():
    """Weights must reproduce the exact area, points must stay inside and
    must differ between draws.

    The last one is the load-bearing property: with a *fixed* rule the
    optimiser drives the quadrature sum to zero while the field diverges --
    Pi/Pi_ref fell to 0.007 over 300 epochs while the von Mises error grew
    from 0.085 to 2.70. See docs/phase2_energy_form.md.
    """
    from geometry.parametric_dogbone import _point_in_dogbone
    from physics.weak_form import EnergyLoss

    fi = get_fillet_geometry(GEOMETRY_DEFAULT)

    def outside_fraction(target, n_draw=8):
        el = EnergyLoss(mu=MU, lam=LAM,
                        stress_state=MATERIAL_CONFIG["state"],
                        resample=True, n_per_elem=3, n_elem_target=target)
        mesh = el._mesh(GEOMETRY_DEFAULT)
        fracs, draws = [], []
        for _ in range(n_draw):
            pts, w = el._stratified(mesh)
            assert abs(w.sum() - mesh.areas.sum()) < 1e-9 * mesh.areas.sum()
            fracs.append(float(
                (~_point_in_dogbone(pts[:, 0], pts[:, 1], fi)).mean()))
            draws.append(pts)
        return float(np.mean(fracs)), draws

    # Points are not all strictly inside: along the fillet the material lies
    # *outside* the arc's circle, so a straight triangle edge between two arc
    # nodes cuts into the void.  The escaped fraction is small and shrinks
    # under refinement -- which is what makes it a discretisation artefact
    # rather than a bug.
    coarse, _ = outside_fraction(400)
    fine, draws = outside_fraction(1400)
    assert fine < 3e-3
    assert fine < coarse

    assert not np.allclose(draws[0], draws[1])
    assert not np.allclose(draws[1], draws[2])


def test_energy_quadrature_converges():
    """The integral is resolved: refining the mesh barely moves Pi.

    Guards the other side of the trade -- the quadrature has to be accurate
    enough that a lower energy means a better field, not a coarser rule.
    """
    from physics.weak_form import EnergyLoss

    torch.manual_seed(0)
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).eval()
    mesh = generate_dogbone(GEOMETRY_DEFAULT, rng=np.random.default_rng(0))
    coll = sample_collocation_points(mesh, n_interior=1,
                                     rng=np.random.default_rng(1))
    bpc = torch.tensor(coll.boundary_pc, dtype=torch.float32).unsqueeze(0)
    x_m = torch.tensor([coll.x_max])
    y_m = torch.tensor([coll.y_max])
    u_d = torch.tensor([1.0])
    with torch.no_grad():
        z = model.encode(bpc, x_m, y_m, sample_ids=torch.tensor([0]))

    el = EnergyLoss(mu=MU, lam=LAM, stress_state=MATERIAL_CONFIG["state"],
                    resample=False, quad_degree=2, n_elem_target=800)
    err = el.integration_error(model, GEOMETRY_DEFAULT, z, u_d, x_m, y_m,
                               refine=2.0)
    assert err["rel_mesh"] < 5e-3
    assert err["rel_rule"] < 5e-3


def test_energy_traction_terms_vanish_on_the_exact_solution():
    """The added traction terms must be redundant, not corrective.

    docs/phase3_transverse_conditioning.md justifies adding P.N = 0 residuals
    back to the energy form on the grounds that they are *natural* BCs of Pi
    and therefore cannot bias the converged answer -- they only reweight the
    approach.  That argument is only sound if the terms actually vanish on the
    solution.  Checked against the finite-element reference: the traction
    residual on the free boundary must be small in the same units the loss
    uses (nondimensionalised by S0), and far below the residual a partly
    trained network carries.
    """
    import numpy as np

    from config import (LOADING_CONFIG, MATERIAL_CONFIG, NONDIM_SCALES,
                        TRAINING_CONFIG, get_fillet_geometry)
    from geometry.banks import VAL_BANK_SEED, build_geometry_bank
    from verification.fem.mesh import build_mesh, default_h
    from verification.fem.solver import solve as fem_solve

    mu, lam = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
    state = MATERIAL_CONFIG["state"]
    bank = build_geometry_bank(3, TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    gmesh, _ = bank[0]
    fi = get_fillet_geometry(gmesh.params)
    mesh = build_mesh(gmesh.params, h=default_h(fi) * 3.0)
    sol = fem_solve(mesh, LOADING_CONFIG["u_max"], mu, lam, state,
                    n_steps=1, verbose=False)

    # Traction on the gauge top, from the reference stress field.
    P = sol.P                                  # [n_elem, 2, 2]
    cx, cy = sol.centroids[:, 0], sol.centroids[:, 1]
    on_gauge = (cx < fi["x_g"]) & (cy > 0.9 * fi["H_gauge"])
    assert on_gauge.sum() > 5, "no elements found under the gauge top"
    # Outward normal there is +y, so the traction is the second column of P.
    t = P[on_gauge][:, :, 1] / NONDIM_SCALES["S0"]
    rms = float(np.sqrt((t ** 2).sum(axis=1).mean()))
    assert rms < 5e-2, (
        f"traction residual on the free boundary is {rms:.3e} in loss units; "
        "the term is not redundant on the solution, so it would bias the "
        "converged answer rather than only reweighting the approach")


def test_bank_parameters_depend_only_on_seed_and_index():
    """A geometry's parameters must not depend on the collocation settings.

    The bank originally drew parameters, meshes and collocation points from
    one shared RNG, so the parameter sequence moved when `n_interior` or
    `n_boundary_per_segment` changed, or when `mesh_only` skipped the mesh
    entirely.  "Validation geometry 11" then named different shapes in
    different runs -- which is how the Phase 2-4 studies came to evaluate on a
    narrow taper band while believing they had picked a spread, and it makes
    every geometry index in the write-ups incomparable across documents.
    """
    import config as cfg
    from config import TRAINING_CONFIG
    from geometry.banks import VAL_BANK_SEED, build_geometry_bank

    ranges = TRAINING_CONFIG["bank_geo_ranges"]
    keys = ("L_total", "W_grip", "W_gauge", "R_fillet")

    def draw(mesh_only, n_interior, n_seg):
        old_i = cfg.COLLOCATION_CONFIG.get("n_interior")
        old_s = cfg.COLLOCATION_CONFIG.get("n_boundary_per_segment")
        cfg.COLLOCATION_CONFIG["n_interior"] = n_interior
        cfg.COLLOCATION_CONFIG["n_boundary_per_segment"] = n_seg
        try:
            bank = build_geometry_bank(4, ranges, holes_on=False,
                                       seed=VAL_BANK_SEED,
                                       mesh_only=mesh_only)
        finally:
            cfg.COLLOCATION_CONFIG["n_interior"] = old_i
            cfg.COLLOCATION_CONFIG["n_boundary_per_segment"] = old_s
        return [[float((p if mesh_only else p[0].params)[k]) for k in keys]
                for p in bank]

    base = draw(True, 1500, 400)
    for label, got in (
            ("a different interior count", draw(True, 400, 400)),
            ("a different boundary count", draw(True, 1500, 120)),
            ("meshes actually built", draw(False, 1500, 400)),
            ("meshes built at another resolution", draw(False, 400, 120)),
    ):
        assert got == base, (
            f"bank parameters changed under {label}; a geometry's shape must "
            "depend only on (seed, index)")
