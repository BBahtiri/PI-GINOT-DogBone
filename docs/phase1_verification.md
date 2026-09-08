# Phase 1 verification report

Four levels, each gating the one above it. Levels 0 and 1 run in the test
suite (`pytest`, ~27 s on CPU) and as standalone reports; level 2 needs
training and is a CLI driver.

Reproduce:

```
python -m verification.operators
python -m verification.mms
python -m verification.geometry_checks
python -m verification.solve_uniaxial --epochs 4000 --device cuda --track
```

---

## Why this exists

The repository had no check of the physics against a known answer. That is a
specific kind of blind spot, not a general one: a training curve cannot see an
error in the operators. If `Div P` has a sign error, or a fillet normal points
inward, or the traction pairs the wrong stress index with the wrong normal
component, the optimiser does not complain — it converges to a different
field, the loss goes down, and every reported stress is quietly wrong.

`verification/analytic_fields.py` is what makes the checks possible: it
presents a closed-form displacement field through the same interface
`PI_GINOT` exposes, so `equilibrium_residual`, `traction` and
`first_piola_kirchhoff_stress` can be driven with an exact answer. Nothing is
reimplemented — the code under test is the code that trains.

---

## Level 0 — physics operators

All in float64. Every figure is a relative error whose exact value is 0.

| check | result |
|---|---|
| rigid translation → P = 0 | 0.000e+00 |
| finite rotation (0.35 rad) → P = 0 | 0.000e+00 |
| objectivity P(QF) = Q P(F), θ = 0.1 … 2.5 rad | ≤ 3.24e-15 |
| affine patch test: max std(P_ij)/&#124;P&#124; | 4.31e-16 |
| affine patch test: max &#124;P_ij(x) − P_ij(F)&#124;/&#124;P&#124; | 0.000e+00 |
| section resultant vs 2·H·P11, four sections | 0.000e+00 |

**The boundary-condition check is the one worth reading.** On the real curved
quarter-model geometry, a laterally-free uniaxial stretch has
P12 = P21 = P22 = 0. With correct outward normals every boundary condition of
the model is then satisfied *exactly*, except on the fillet arc:

| segment | ‖P·N‖/E |
|---|---|
| bottom | 0.000e+00 |
| right_grip | 0.000e+00 |
| gauge_top | 0.000e+00 |
| left_symmetry | 0.000e+00 |
| right_arc | 3.93e-02 |

The four zeros check the normals, the segment tagging and the traction index
pairing together, on the actual domain. The arc must be non-zero — a uniform
stretch is not the dog-bone solution — so a zero there would mean the arc
normals were dead rather than correct.

The patch test deliberately does **not** check `Div P`. For an affine field
the stress is a constant with no autograd graph, so a divergence check would
be testing the constant-stress shortcut in `equilibrium_residual` rather than
the differentiation chain. That is level 1's job.

---

## Level 1 — method of manufactured solutions

Manufactured field on the dog-bone domain:

```
u(X, Y) = a · sin(πX/L_half) · (Y/H_grip)
v(X, Y) = b · (X/L_half) · (Y/H_grip)²
```

`P*` and `Div P*` are derived with sympy from the **closed-form Lambert-W
solution** of the plane-stress closure, not from this repository's Newton
solve, so a defect in the torch implementation cannot cancel itself out. The
symbolic reference is itself checked first: the Lambert-W form satisfies the
closure residual to **3.68e-16** over J₂D ∈ [0.4, 2.5].

Interior — `Div P` from `equilibrium_residual` against symbolic `Div P*`:

| state | J₂D range | ‖Div P*‖ rms | rel L2 | rel L∞ |
|---|---|---|---|---|
| plane strain | [1.0000, 1.0217] | 2.25e+00 | 2.66e-15 | 6.52e-15 |
| plane stress | [1.0000, 1.0217] | 2.05e+00 | 2.88e-15 | 6.71e-15 |

Severity sweep (plane stress) — the operator over the whole deformation range,
well past anything the J barrier would permit:

| amplitude (a, b) mm | J₂D range | ‖Div P*‖ rms | rel L2 | rel L∞ |
|---|---|---|---|---|
| (0.35, 0.25) | [1.0000, 1.0219] | 2.04e+00 | 2.85e-15 | 6.08e-15 |
| (1.5, 1.0) | [0.9906, 1.0931] | 7.90e+00 | 2.20e-15 | 6.11e-15 |
| (4.0, 3.0) | [0.8676, 1.2553] | 2.13e+01 | 2.77e-15 | 1.31e-14 |
| (8.0, 6.0) | [0.2225, 1.5184] | 5.42e+01 | 2.35e-14 | 3.51e-14 |

Boundary — `P·N` against symbolic `P*·N`, relative L∞ per segment:

| state | bottom | right_grip | right_arc | gauge_top | left_symmetry |
|---|---|---|---|---|---|
| plane strain | 3.53e-15 | 1.20e-14 | 1.09e-14 | 1.43e-14 | 1.29e-14 |
| plane stress | 6.33e-15 | 1.40e-14 | 1.31e-14 | 1.48e-14 | 1.76e-14 |

The plan's tolerances here were rel L2 < 1% on a rectangle and < 2% on a
curved domain. Those are *solver* tolerances. Verified as an operator, the
answer is machine precision on the curved domain directly.

---

## Level 0 (geometry) — normals, and a measure-consistency defect

MMS compares `P·N` using the mesh's own normals on both sides, so a normal
pointing the wrong way cancels out and cannot be seen. This level compares the
geometry itself against its closed form, over the 24-geometry validation bank
plus the default and two corner cases (the shallowest and deepest fillet the
parameter ranges allow).

**The boundary description is exact:**

| check | worst over 27 geometries |
|---|---|
| points lie on the analytic curve (rel. to L_half) | 5.52e-08 |
| normals are unit length | 1.21e-07 |
| normals match the closed form | 1.22e-07 |
| segment length vs closed form | 6.19e-07 |
| boundary loop closes | 0.000e+00 |
| points outside after a +N step | 0 |
| points inside after a −N step | 0 |

The 1e-7 floor is float32 storage of the boundary arrays, not a geometric
error. Outwardness is tested against `_point_in_dogbone`, the same predicate
the interior sampler uses.

**Measure consistency is not.** `generate_dogbone` places
`n_pts_per_segment` on every segment regardless of its length. At the shipped
settings the 5 mm symmetry face carries the same 400 points as the 27 mm
bottom face:

- the encoder's boundary point cloud varies by up to **17.5×** in points per
  millimetre across segments, so what the operator sees of the shape is
  weighted by segment count rather than by arc length;
- the collocation sampler's pool is starved on the longest segment, so it
  cannot deliver that segment's arc-length share. It returns as few as
  **72.4%** of the `n_total_boundary` points requested, and lands up to
  **11.3 percentage points** from a proportional split;
- that also mis-weights `L_part`, which pools `bottom`, `left_symmetry` and
  `right_grip` into a single mean, so the segments' relative influence follows
  their point counts.

`COLLOCATION_CONFIG["boundary_measure_consistent"]` shares the identical point
budget out by arc length instead. Worst case over the same 27 geometries:

| | off (default) | on |
|---|---|---|
| encoder PC density max/min | 17.45× | 1.41× |
| collocation vs arc-length split | 11.31 pp | 0.66 pp |
| requested boundary points not returned | 27.6% | 0.0% |

Default stays `False` so existing checkpoints and results remain
interpretable. It changes the encoder's input representation, so it belongs in
the ablation matrix rather than being switched on silently.

---

## Level 2 — solver, against a closed-form solution

A near-prismatic specimen (`W_gauge = 19.99` vs `W_grip = 20`, so the fillet
spans 0.28 mm of a 27 mm half-length) makes the decoder's hard boundary
conditions exactly those of a uniaxial test. The homogeneous field
`u = (λ₁−1)X`, `v = (β−1)Y` then satisfies every boundary condition *and*
equilibrium exactly, so it **is** the solution rather than an approximation to
it — any difference is solver error measured against a number.

Load `u_δ = 1.0` mm → `λ₁ = 1.03703704`, `β = 0.99163273`,
`P₁₁ = 27.440537` MPa.

4000 epochs, Adam at 3e-4, 800 interior and 500 boundary collocation points,
CPU, ~63 min. Shipped loss weights, unchanged.

| metric | untrained | trained | tolerance | |
|---|---|---|---|---|
| u relative L2 | 5.21e-06 | **1.89e-06** | 1e-03 | PASS |
| v relative L2 | 9.99e-01 | **2.48e-04** | 1e-03 | PASS |
| P₁₁ relative max error | 5.45e-02 | **2.51e-05** | 1e-03 | PASS |
| std(P₁₁)/mean(P₁₁) | 7.14e-05 | 6.90e-06 | — | — |
| &#124;P₂₂&#124;/E | 8.83e-03 | 2.33e-06 | — | — |
| &#124;P₁₂&#124;/E | 8.49e-06 | 7.11e-08 | — | — |
| &#124;P₃₃&#124;/E | 2.24e-16 | 2.24e-16 | 1e-08 | PASS |

Mean P₁₁: 28.9259 → **27.4400** MPa against an exact 27.4405.

Reading the rows:

- **v is the hard one.** The hard boundary conditions give u for free
  (`u = u_δ·ξ + ξ(1−ξ)φ_u` already has the right endpoints), so u starts at
  5e-6 and barely improves. The transverse field starts at essentially zero
  and has to be learned entirely from the traction-free condition on the
  lateral faces — it goes from 100% error to 2.5e-4. That is the row that
  says the solver works.
- **The untrained P₁₁ is 5.4% high** because the initial field is a uniform
  axial stretch with no lateral contraction, i.e. laterally constrained. It
  converges to the laterally-free value.
- **std(P₁₁)/mean(P₁₁) reaches 6.9e-6**, within a factor of seven of the
  finite-element patch-test tolerance of 1e-6, which gradient descent has no
  reason to reach at all.
- **The loss plateaus at 8.52e-06** while `L_eq` is 1e-11 and `L_top` is
  5e-12. The remainder is `w_trac_arc · L_trac_arc` on the 0.28 mm residual
  fillet, where the exact solution genuinely is not homogeneous. The plateau
  is the specimen's departure from a perfect rectangle, not solver
  stagnation — which is also why the field errors sit at 1e-6 rather than
  falling further.

**Phase 1 gate: passed.** The operators are exact and the solver recovers a
known solution to within 40–500× of the required tolerances.

---

## What is not covered

- **Stress concentration against a reference (plan case 1.4).** The plan
  suggested Kirsch's plate with a circular hole, K_t = 3. This operator has no
  hole topology — holes are permanently disabled — and a filleted dog-bone has
  no closed-form K_t. A meaningful concentration check needs an external FEM
  reference field, which nothing in the repository can currently produce. This
  is the one Phase 1 case still open, and it is the one that most directly
  gates the paper's headline claim about local stress fidelity.
- **The solver on the real dog-bone.** Level 2 verifies the solver where the
  answer is known, which is a prismatic bar. That the solver finds the right
  answer there does not by itself establish accuracy on a steep fillet.
