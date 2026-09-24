#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-strain plane-stress finite-element solver for the quarter dog-bone.

Purpose
-------
An *independent discretisation* of the same boundary-value problem the
operator is trained on, so that "how accurate is the predicted stress at the
fillet" has an answer.  Until now the project had no ground truth at all: the
only physics numbers available were residuals, which say a field is
self-consistent, not that it is right.

What is and is not independent
------------------------------
Independent: the discretisation (Galerkin finite elements on a conforming
triangulation, versus pointwise collocation of the strong form), the solution
method (Newton-Raphson on the assembled residual, versus gradient descent on
a penalty sum), the boundary-condition treatment (elimination, versus hard
substitution plus penalties) and the quadrature.

Shared: the constitutive law, ``physics.neo_hookean``.  That is deliberate.
The law is already verified independently to machine precision against the
Lambert-W closed form and a sympy-derived manufactured solution (levels 0 and
1), so re-deriving it here would add risk, not confidence.  What this solver
tests is the *solution* -- and a discretisation error cannot hide behind a
constitutive error that has already been ruled out.

Element
-------
Constant-strain triangle.  F is constant per element, so the plane-stress
closure applies exactly once per element rather than being averaged over a
quadrature rule, and the element integral is exact.  Stress is therefore
piecewise constant and first-order accurate, which is why every reference
produced here ships with a mesh-convergence error bar rather than a bare
number.

Boundary conditions match what the decoder hard-enforces:
    u_x = 0        on x = 0        (left_symmetry)
    u_x = u_delta  on x = L_half   (right_grip)
    u_y = 0        on y = 0        (bottom)
everything else free, i.e. traction-free on the fillet arc and the gauge top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from physics.neo_hookean import first_piola_kirchhoff_stress
from verification.fem.mesh import FemMesh

DTYPE = torch.float64


@dataclass
class FemSolution:
    """Converged displacement field and the diagnostics that qualify it."""
    mesh: FemMesh
    u: np.ndarray                      # (n_node, 2)
    u_delta: float
    P: np.ndarray                      # (n_elem, 2, 2) element stress
    J2D: np.ndarray                    # (n_elem,)
    residual_history: list = field(default_factory=list)
    n_newton: int = 0
    wall_time: float = 0.0

    @property
    def centroids(self) -> np.ndarray:
        return self.mesh.nodes[self.mesh.tris].mean(axis=1)

    def von_mises(self) -> np.ndarray:
        """Element von Mises stress from the Cauchy stress (plane stress)."""
        F = deformation_gradients(self.mesh, self.u)
        J3D = self.J2D * _F33(self.J2D)
        s = np.einsum("eij,ekj->eik", self.P, F) / J3D[:, None, None]
        s11, s22, s12 = s[:, 0, 0], s[:, 1, 1], 0.5 * (s[:, 0, 1] + s[:, 1, 0])
        return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3.0 * s12 ** 2)

    def axial_resultant(self, x: float, n_y: int = 4000) -> float:
        """N(x) = 2 * integral of P11 over the reference section."""
        from verification.fem.interpolate import sample_elementwise
        from physics.uniaxial import section_half_height
        h = float(section_half_height(x, self.mesh.fillet_info))
        y = np.linspace(0.0, h, n_y + 1)
        y = 0.5 * (y[:-1] + y[1:])
        pts = np.stack([np.full(n_y, x), y], -1)
        p11 = sample_elementwise(self.mesh, self.P[:, 0, 0], pts)
        return float(2.0 * (h / n_y) * np.nansum(p11))


def _F33(J2D: np.ndarray) -> np.ndarray:
    from config import MATERIAL_CONFIG
    from physics.neo_hookean import solve_F33_plane_stress
    t = torch.tensor(J2D, dtype=DTYPE).reshape(1, -1, 1)
    return solve_F33_plane_stress(t, MATERIAL_CONFIG["mu"],
                                  MATERIAL_CONFIG["lam"]).numpy().ravel()


def deformation_gradients(mesh: FemMesh, u: np.ndarray) -> np.ndarray:
    """F = I + sum_a u_a (x) grad(N_a), constant per element."""
    ue = u[mesh.tris]                              # (n_elem, 3, 2)
    F = np.einsum("eai,eaj->eij", ue, mesh.grads)  # (n_elem, 2, 2)
    F[:, 0, 0] += 1.0
    F[:, 1, 1] += 1.0
    return F


def constitutive(F: np.ndarray, mu: float, lam: float, state: str,
                 want_tangent: bool = True):
    """Element stress and, optionally, the tangent dP/dF.

    The tangent is taken by autograd through the same
    ``first_piola_kirchhoff_stress`` the training loss uses -- including the
    plane-stress closure's own derivative -- so the Newton iteration is
    consistent with the constitutive law rather than with an approximation of
    it.  Four batched backward passes, one per component of P.
    """
    Ft = torch.tensor(F, dtype=DTYPE, requires_grad=want_tangent)
    g = [Ft[:, 0, 0].reshape(1, -1, 1) - 1.0,
         Ft[:, 0, 1].reshape(1, -1, 1),
         Ft[:, 1, 0].reshape(1, -1, 1),
         Ft[:, 1, 1].reshape(1, -1, 1) - 1.0]
    P11, P12, P21, P22, J2D = first_piola_kirchhoff_stress(
        *g, mu, lam, state)
    comps = [P11, P12, P21, P22]
    P_np = np.stack([c.detach().numpy().ravel() for c in comps],
                    axis=-1).reshape(-1, 2, 2)
    J_np = J2D.detach().numpy().ravel()
    if not want_tangent:
        return P_np, J_np, None

    A = np.empty((len(F), 2, 2, 2, 2))
    for n, c in enumerate(comps):
        i, j = divmod(n, 2)
        gr = torch.autograd.grad(c.sum(), Ft, retain_graph=(n < 3))[0]
        A[:, i, j, :, :] = gr.numpy()
    return P_np, J_np, A


def _dirichlet(mesh: FemMesh, u_delta: float):
    """Constrained DOF indices and their prescribed values."""
    n = mesh.n_node
    dofs, vals = [], []
    for i in mesh.boundary["left_symmetry"]:
        dofs.append(2 * i + 0); vals.append(0.0)
    for i in mesh.boundary["right_grip"]:
        dofs.append(2 * i + 0); vals.append(u_delta)
    for i in mesh.boundary["bottom"]:
        dofs.append(2 * i + 1); vals.append(0.0)
    dofs = np.asarray(dofs, dtype=np.int64)
    vals = np.asarray(vals, dtype=float)
    # A node can appear twice (corners); keep the last, they agree by
    # construction since the corner values are all 0 or u_delta consistently.
    uniq, idx = np.unique(dofs, return_index=True)
    return uniq, vals[idx]


def assemble(mesh: FemMesh, u: np.ndarray, mu: float, lam: float, state: str,
             want_tangent: bool = True):
    """Global internal force and tangent stiffness."""
    F = deformation_gradients(mesh, u)
    P, J2D, A = constitutive(F, mu, lam, state, want_tangent)

    # f_ai = A_e * P_iJ * grad_aJ
    fe = np.einsum("e,eij,eaj->eai", mesh.areas, P, mesh.grads)
    f = np.zeros(2 * mesh.n_node)
    np.add.at(f, (2 * mesh.tris[:, :, None] + np.arange(2)).ravel(), fe.ravel())

    if not want_tangent:
        return f, None, P, J2D

    # K_{ai,bk} = A_e * A_iJkL * grad_aJ * grad_bL
    Ke = np.einsum("e,eijkl,eaj,ebl->eaibk", mesh.areas, A, mesh.grads,
                   mesh.grads)
    n_e = mesh.n_elem
    gdof = (2 * mesh.tris[:, :, None] + np.arange(2)).reshape(n_e, 6)
    rows = np.repeat(gdof, 6, axis=1).ravel()
    cols = np.tile(gdof, (1, 6)).ravel()
    K = sp.coo_matrix((Ke.reshape(n_e, 36).ravel(), (rows, cols)),
                      shape=(2 * mesh.n_node, 2 * mesh.n_node)).tocsr()
    return f, K, P, J2D


def newton(mesh: FemMesh, con: np.ndarray, val_of_step, u0: np.ndarray,
           mu: float, lam: float, state: str, n_steps: int = 1,
           tol: float = 1e-10, max_iter: int = 30, verbose: bool = True,
           max_backtrack: int = 25):
    """Newton core: load-stepped, line-searched, arbitrary Dirichlet set.

    Args:
        con: constrained DOF indices.
        val_of_step: callable(load_fraction) -> prescribed values for ``con``.
        u0: initial guess, (n_node, 2).

    Returns (u, history, n_iter).
    """
    n_dof = 2 * mesh.n_node
    scale = mu * mesh.areas.sum()
    free = np.setdiff1d(np.arange(n_dof), con)
    u = u0.copy()
    history = []
    total_iter = 0

    def residual(uu):
        f, _, _, _ = assemble(mesh, uu, mu, lam, state, want_tangent=False)
        return np.linalg.norm(f[free]) / scale

    def min_detF(uu):
        F = deformation_gradients(mesh, uu)
        return float((F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]).min())

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        flat = u.ravel().copy()
        flat[con] = val_of_step(frac)
        u = flat.reshape(-1, 2)

        for it in range(max_iter):
            f, K, P, J2D = assemble(mesh, u, mu, lam, state)
            r = np.linalg.norm(f[free]) / scale
            history.append(r)
            if verbose:
                print(f"    step {step}/{n_steps}  iter {it:2d}  "
                      f"|r|/(mu*A) = {r:.3e}", flush=True)
            if r < tol:
                break

            du = spla.spsolve(K[free][:, free].tocsc(), -f[free])

            alpha = 1.0
            base = u.ravel().copy()
            for _ in range(max_backtrack):
                trial = base.copy()
                trial[free] += alpha * du
                uu = trial.reshape(-1, 2)
                if min_detF(uu) > 1e-6 and residual(uu) < r:
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(
                    f"line search failed at step {step}, iter {it}: no step "
                    f"length reduces the residual from {r:.3e}")
            u = uu
            total_iter += 1
        else:
            raise RuntimeError(
                f"Newton failed to converge at step {step}: |r| = {r:.3e}")
    return u, history, total_iter


def solve(mesh: FemMesh, u_delta: float, mu: float, lam: float, state: str,
          n_steps: int = 3, tol: float = 1e-10, max_iter: int = 30,
          verbose: bool = True, max_backtrack: int = 25) -> FemSolution:
    """Newton-Raphson with load stepping and a backtracking line search.

    Two details matter for robustness, both learned the hard way:

    *Initial guess.*  Starting from u = 0 with the grip displacement applied
    only on the loaded edge is a discontinuous state: the elements touching
    the grip see F11 of order 1 + u_delta/h, which at the refined fillet
    spacing is about 4.  The tangent there is nowhere near representative and
    the first Newton step overshoots into inverted elements.  Seeding with the
    affine field u_x = u_delta * X / L_half -- the same baseline the decoder's
    hard boundary conditions build in -- starts the iteration in the right
    basin.

    *Line search.*  A full Newton step is still allowed to drive J2D
    negative, which is not merely inaccurate but outside the constitutive
    law's domain (ln J).  The step is halved until the residual decreases and
    every element keeps a positive determinant.

    Convergence is measured on the free-DOF residual normalised by mu times
    the domain area, so the tolerance means the same thing on every mesh.
    """
    t0 = time.time()
    L_half = mesh.fillet_info["L_half"]

    # Affine seed, matching the decoder's hard-BC baseline.
    u0 = np.zeros((mesh.n_node, 2))
    u0[:, 0] = u_delta * mesh.nodes[:, 0] / L_half

    con, _ = _dirichlet(mesh, u_delta)
    val_of_step = lambda frac: _dirichlet(mesh, u_delta * frac)[1]

    u, history, total_iter = newton(
        mesh, con, val_of_step, u0, mu, lam, state, n_steps=n_steps,
        tol=tol, max_iter=max_iter, verbose=verbose,
        max_backtrack=max_backtrack)

    f, _, P, J2D = assemble(mesh, u, mu, lam, state, want_tangent=False)
    return FemSolution(mesh=mesh, u=u, u_delta=u_delta, P=P, J2D=J2D,
                       residual_history=history, n_newton=total_iter,
                       wall_time=time.time() - t0)
