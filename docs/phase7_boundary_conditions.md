# Phase 7 — boundary conditions: are they right, and are they the error?

**Status.** Audit and literature review complete.
- Phase A (evaluation-only) and Phase B1 (FEM-only) were specified with their
  predictions and decision rules and committed (`5f06482`, scripts `b875fef`)
  **before either was run**. Both have now run; results are in §8.
- Phase C (training) is gated on them; nothing in it has been run.

---

## In plain words

A boundary condition is the rule for what happens at the edges of the specimen.
There are two kinds. **Dirichlet** conditions fix *where the material is*: the
grip end is pulled by exactly 1 mm, and the two symmetry cuts don't move across
themselves. **Natural** conditions fix *what forces act*: the curved edge of the
fillet is free, so nothing pushes on it.

- **The Dirichlet side is correct.** The neural network and the finite-element
  reference impose exactly the same fixed-displacement rules, at the same
  places, with the same values. The network builds them into its formula, so
  they hold exactly, not approximately. The way it does this matches what the
  current research recommends.
- **The natural side is where to look.** In the energy method, "the curved edge
  is free" is never imposed directly. It only comes out right if training finds
  the true minimum. The peak stress sits *on* that curved edge, and several
  recent papers find that energy-based networks lose accuracy exactly there.
- **One modelling caveat, shared by both solvers.** The specimen is pulled at
  the end of the fillet, with no straight grip section. The peak is only about
  one grip half-width away from where the load is applied. This is not an error
  relative to FEM, because FEM makes the same choice. But it may change what
  the `K_t` number means for a real test specimen.

The plan measures these cheaply, before any retraining.

**What the measurements found** *(added after the run; details in §8)*:

- **The Dirichlet side is exact.** The error is zero on every model and
  geometry.
- **The free curved edge is not quite free.** The network leaves a small push
  on it, about 3× more than the FEM does.
  - At the peak this makes the predicted stress read 0.4–0.9% low. That is
    comparable in size to the `K_t` error itself (1.0–1.9%).
  - But removing it would shrink that error by only 12–22%. This is a
    first-order, exploratory estimate. The rest of the error does not point the
    same way on every geometry.
- **The grip model changes the true answer.** Switching the grip from
  frictionless to clamped moves the reference `K_t` itself:
  - by 1.2–4.2% in range (mean 2.3%, above the network's mean error of 1.5%);
  - by about 4.7% out of range.

  The grip face sits close to the fillet.

---

## 1. The audit — what each solver imposes

Quarter model: symmetry about `x = 0` and `y = 0`, grip face at `x = L_half`,
free edge = gauge top + fillet arc. Every row was checked in code.

| boundary | component | intended | PINN (`models/physics_decoder.py::_apply_hard_bc`) | FEM (`verification/fem/solver.py::_dirichlet`) | match |
|---|---|---|---|---|---|
| `left_symmetry` `x = 0` | `u` | 0 | hard: factor `ξ = x/L_half` | eliminated | ✓ |
| | shear `P₂₁` | 0 | natural | natural | ✓ |
| `bottom` `y = 0` | `v` | 0 | hard: factor `η = y/H_grip` | eliminated | ✓ |
| | shear `P₁₂` | 0 | natural | natural | ✓ |
| `right_grip` `x = L_half` | `u` | `u_δ` = 1 mm | hard: lift `u_δ·ξ`, factor `(1 − ξ)` | eliminated | ✓ |
| | `v` / shear `P₂₁` | free / 0 ("frictionless grip") | natural | natural | ✓ |
| `gauge_top`, `right_arc` | `P·N` | 0 | natural | natural | ✓ |

The decoder ansatz is `u = u_δ ξ + ξ(1 − ξ) φ_u` and `v = η φ_v`.

### Dirichlet findings

- **D1 — same sets, same values, same places.** Both solvers constrain the same
  three edges (`u = 0`, `u = u_δ`, `v = 0`) with the same `u_δ =
  LOADING_CONFIG["u_max"]`. The trainer and `compare_reference` both read it
  there, with no ramp.
- **D2 — the constrained location is the real end face.** `x_max` and `y_max`
  are the maxima of the geometry's own boundary nodes. `_sample_line` includes
  the endpoints, so `x_max = L_half` and `y_max = H_grip`, to float32 storage.
- **D3 — the multipliers are the kind the literature recommends (§2.1–2.2).**
  - Each component's multiplier vanishes only where that component is
    prescribed.
  - Each multiplier is first-order normalised, up to a constant: `x(L − x)/L`
    has `∂/∂n = ±1` on both faces, and `y` has `∂/∂n = 1` on `y = 0`.
  - Both are smooth.
  - No multiplier is a product of distance functions whose zero sets
    *intersect*. The two zero sets of `ξ(1 − ξ)` are parallel and never meet.
    Intersecting zero sets are the failure mode Berrone et al. (2023) found can
    ruin convergence.
- **D4 — the one alternative multiplier was already tested.** Phase 3 measured
  the local-height form `v = (y/h(x)) φ_v` on the FEM solution, with no
  training. It is *worse*: demanded dynamic range 4.15 vs 2.02.
- **D5 — minor, mixed model only.**
  - In `physics/mixed.py` the potential `φ₂` carries `ξ(1 − η)`. Its zero sets
    meet at `(0, H_gauge)`: an un-normalised product at a corner, far from the
    fillet.
  - Its `η = y/h(x)` is normalised only up to `√(1 + h′²)`. That factor varies
    along the arc: about 1.7 on the default geometry at the grip corner.
  - Not the shipped configuration.

### Natural-condition findings

- **N1 — every non-Dirichlet condition is natural in both solvers.** That is
  the free edge, both symmetry-plane shears and the grip-face shear. A natural
  condition holds exactly only at the exact minimiser.
  - The FEM approaches that minimiser by refining the mesh. Its constant-strain
    triangles give piecewise-constant stress, so the traction at a boundary
    element is `O(h)`.
  - The network approaches it only as well as the optimiser converges.
- **N2 — the peak lives on a natural boundary.** The reference peak sits on the
  arc just past tangency (`x*/x_g ≈ 1.03–1.11`). **No free-edge traction
  residual has ever been measured for the energy form.**
  - The `σ₂₂/peak` column in Phases 1–2 is the whole-domain RMS *error* of `σ₂₂`
    against the FEM, not a boundary traction: 5.4e-3 single-geometry, 5.9e-2
    for the Phase 2 point-cloud operator.
  - Case 1.4's "σ₂₂ at 30% of peak" was the legacy penalty model's `σ₂₂` error,
    in a gauge where the reference is ≈ 0.
- **N3 — a penalty lever exists but was never scored on `K_t`.** `EnergyLoss`
  already has `w_trac_top` and `w_trac_arc`, both default 0. Phase 3 swept only
  `w_trac_top`, and scored only `v` and L2: `v` moved ±10%, von Mises got 3.5×
  worse. `K_t` did not exist as a metric until Phase 6.1.
  - The arc term was never swept. Phase 3's own write-up says why: the reason
    was "an inherited assumption, not something re-checked here".

### Grip modelling (physics, not solver error)

- **G1 — the load is applied at the end of the fillet.** The arc runs straight
  to the grip face; there is no prismatic grip section. On the Phase 6.7 sets,
  the distance from the grip face to the tangency point is
  - `dx / H_grip = 0.76–1.21` in range (median 1.06);
  - `0.69–0.90` out of range.

  Saint-Venant end effects decay over about one specimen width (Horgan 1989),
  here `2 H_grip`. **The fillet peak is inside the end-effect zone.** Both
  solvers share this, so it cannot create PINN-vs-FEM error. It does make `K_t`
  the concentration factor of *this loading*, not the textbook shoulder-fillet
  `K_t` under remote tension (Noda & Takase 2004).
- **G2 — the free-`v` grip avoids a corner singularity.** The grip face meets
  the arc at an interior angle of 22–53° in range and 56–67° out of range.
  - A roller edge (`u` fixed, shear-free) meeting a free edge is equivalent, by
    reflection, to a free–free wedge of twice the angle. That stays non-singular
    below 180°.
  - A *clamped* edge meeting a free edge becomes singular above a threshold
    angle (Williams 1952). Clamping `v = 0` would therefore introduce a weak
    singularity at the out-of-range corners.

### Paper note

The public preprint (Dean & Bahtiri, arXiv:2607.23299) describes the earlier
formulation:
- displacement BCs enforced exactly, as now;
- traction-free and symmetry conditions as trained constraints.

Under the energy form those constraints are natural BCs, and the methods
section needs to say so when it is revised.

---

## 2. What the literature says (2018–2026)

Every citation below was checked against a publisher or arXiv page.

### 2.1 Enforcing Dirichlet conditions

- **Exact imposition through a trial function is the reference approach.**
  - Lagaris et al. (1998) introduced `u = A + F(x, N)`.
  - Berg & Nyström (2018) fit the lift `G` and the distance `D` with networks
    for complex domains.
  - Sukumar & Srivastava (2022) made it systematic. They build approximate
    distance functions (ADFs) from R-functions, normalised so that `φ = 0` and
    `∂φ/∂n = 1` on the boundary. Transfinite interpolation handles non-constant
    data. The result consistently beat penalty PINNs.
- **Head-to-head.** Berrone, Canuto, Pintore & Sukumar (2023) compared four
  methods: penalty, normalised ADF, un-normalised product of distances, and
  Nitsche.
  - The **normalised ADF was the most efficient and accurate**.
  - The un-normalised product "tends to lead to suboptimal results and may even
    ruin the convergence". Its values become excessively small where boundary
    segments meet.
  - The penalty needs an expensive tuning phase.
  - Nitsche "leads to suboptimal solvers".
- **Theory.** An L² boundary penalty costs regularity in the error estimate, and
  an exact ansatz restores it:
  - Müller & Zeinhofer (2022a, b) for Deep Ritz and residual minimisation;
  - Zeinhofer, Masri & Mardal (2025) extend the result to elasticity.
  - In Deep Ritz the penalty's consistency error is `O(1/λ)`, and `λ` must grow
    with the network.
- **Nitsche for networks.** The Deep Nitsche Method (Liao & Ming 2021) is
  consistent and needs no large penalty. In practice it performs like a
  well-tuned penalty (Berrone et al. 2023).
- **Augmented Lagrangian.**
  - hPINN: Lu et al. (2021).
  - PECANN: Basir & Senocak (2022).
  - AL for variational problems: Huang, Wang & Zhou (2022).
  - AL-PINNs: Son et al. (2023).
  - Rowan, Maute, Hampleman & Doostan (2025) compared penalty, learning-rate
    annealing, SA-PINN, Lagrange multipliers and AL on 3D problems. They
    preferred AL, but **excluded distance functions**, because they need heavy
    pre-processing and don't extend to Neumann boundaries on complex 3D domains.
  - Inside the deep energy method, Ouyang et al. (2026) found an AL-type
    "generalised multiplier" more accurate and stable than penalty, L1 exact
    penalty or plain multipliers.
- **Loss balancing (soft methods only).**
  - Gradient-statistics weights: Wang, Teng & Perdikaris (2021).
  - NTK weights: Wang, Yu & Perdikaris (2022).
  - Self-adaptive weights: McClenny & Braga-Neto (2023).

  These reduce the imbalance between the loss terms but do not remove the
  penalty's consistency error.

**Consensus.** When the Dirichlet boundary is simple enough to write a
normalised distance function, exact imposition is the most accurate option. AL
is the best fallback when it is not. Ours is the simple case: three straight
edges, constant data.

### 2.2 When hard constraints hurt

- **Un-normalised products at corners.** They slow or break convergence
  (Berrone et al. 2023). The fix is the R-equivalence normalised ADF.
- **Corner singularities of the ADF.** Its Laplacian blows up at polygon
  vertices (Sukumar & Srivastava 2022). The fix is Wachspress-based transfinite
  lifts (Sukumar & Roy 2026). This matters little here, because the energy form
  needs only `∇u`.
- **Spectral collapse.** Xie et al. (2025) show the boundary function acts as a
  filter on the network's NTK spectrum. "Widely used boundary functions can
  inadvertently induce spectral collapse, leading to optimization stagnation
  despite exact boundary satisfaction." They favour smooth, low-order,
  symmetric multipliers.
- **Strong Neumann imposition on piecewise boundaries.** It is unstable on
  boundaries that are only C⁰ globally (Göschel, Götschel & Ruprecht 2025). Our
  free edge (gauge top + arc) is C¹ at tangency. Its two ends are corners, one
  on the symmetry plane and one at the grip.

### 2.3 Where energy methods lose accuracy

- **Fuhg & Bouklas (2022), *mixed* DEM, finite-strain hyperelasticity.**
  - DEM follows the global field but misses stress concentrations. Only the
    mixed form resolved them "on comparable levels to FEM", and only it resolved
    the stresses near the clamping.
  - Stress outputs with the traction imposed exactly made Neumann conditions
    "approximated more accurately".
  - Our Phase 6.6 construction is the equilibrated, finite-strain relative of
    this.
- **Ye et al. (2026), linear plates with holes.** Relative error at the hole tip
  was 40% for plain DEM, 8.7% for a weighted strong form, and 0.98% for DEM plus
  an equilibrium term, Delaunay integration and local refinement.
- **Liu, Cai & Ramani (2023), Deep Ritz for elasticity.**
  - They impose Dirichlet data by an `H^{1/2}` penalty.
  - Recovering the FEA stress concentration factor needed quadrature adapted to
    the concentration.
- **DCEM (Wang, Sun, Rabczuk & Liu 2024).** Airy/Prandtl stress functions make
  equilibrium and tractions exact. Plate-with-hole von Mises error was 0.020,
  against 0.048 for DEM.
- **Rao, Sun & Liu (2021), mixed PINN.** Tractions imposed through the ansatz;
  soft constraints gave errors near the boundaries.

**Common thread.** The loss sits at free edges and concentrations, and the
remedies are:
- exact tractions, through mixed or stress-function outputs;
- quadrature at the concentration;
- sometimes an added equilibrium term. Our Phase 6.5 showed this can collapse
  unless the stress is a separate output.

### 2.4 Neural operators

- **BOON (Saad et al. 2023).** Corrects the operator kernel so boundary
  conditions hold exactly, for 2–20× gains. Tested on fixed simple domains only.
- **Physics-informed operators on varying geometry use soft boundary conditions
  or none.**
  - PI-DeepONet (Wang, Wang & Perdikaris 2021): soft penalties.
  - PI-GANO (Zhong & Meidani 2025): soft penalties.
  - GINOT (Liu et al. 2026) and GINO (Li et al. 2023): data-trained, with no
    boundary-condition mechanism.
- **Exact imposition on varying domains is recent.**
  - A learned wall distance: GAPINN (Oldenburg et al. 2022).
  - A level-set lift: WINO (Zhu, Wang, Zhang & Rabczuk 2026, hyperelasticity).
  - Strong ADF or projection enforcement in physics-informed operators
    (Göschel et al. 2025), which found it more accurate than weak enforcement.

Our geometry-parameterised hard ansatz is on the exact-imposition side of this
line.

### 2.5 Grip modelling and `K_t`

- **Stress singularities.** A clamped edge meeting a free edge is singular
  above a threshold angle (Williams 1952).
- **End effects.** They decay over about one width in isotropic material
  (Horgan 1989).
- **Formula benchmarks.**
  - Flat shoulder fillets under remote tension: `K_t` formulas accurate to about
    1% (Noda & Takase 2004).
  - ASTM D638 and E8: careful FE `K_t` values (Garrell et al. 2003; Kardak,
    Bilich & Sinclair 2017).
- **A gap.** No study found quantifies how much fillet `K_t` changes between
  end-face and clamped-grip idealisations. Phase B1 measures it for this family.

### 2.6 What this implies here

1. The Dirichlet construction already has the properties the literature asks
   for (D3). Replacing it with penalty, Nitsche or AL is predicted to be worse
   or equal. It is worth running only as a reviewer-facing control.
2. The literature's failure mode for energy methods is the one our
   concentration lives in: a **natural** boundary at a stress concentration.
   That is where to measure first.
3. The grip idealisation (G1, G2) is a modelling question the literature has not
   answered for dog-bones. Phase B answers it with FEM alone.

---

## 3. Hypotheses

| | hypothesis | prior | tested by |
|---|---|---|---|
| H1 | a Dirichlet condition is implemented wrongly | very low (D1–D2) | A1 |
| H2 | the Dirichlet construction (multipliers, lift) costs accuracy | low (D3, D4) | A4, A5 → C4–C5 |
| H3 | the free-edge natural condition is violated enough to matter for `K_t` | moderate (N1–N3, §2.3) | A2, A3 → C1–C3 |
| H4 | the error forms a boundary layer at one boundary | open | A4 |
| H5 | the grip idealisation changes `K_t` at the level of the operator's error | moderate (G1) | B1 |

---

## 4. Phase A — measure (evaluation-only, existing checkpoints)

**Models.** The three Phase 6.7 energy seeds (`fixed_long/oracle-energy`,
`retest/energy_s1`, `retest/energy_s2`).

**Sets.** The 12 in-range (IR) and 8 out-of-range (OOR) geometries of Phase 6.7,
with their cached FEM references at `h_factor` 1.3.

**Scale.** `σ_n` is the reference's area-weighted gauge-mean von Mises: the
`K_t` denominator. Boundary points are 400 per segment; the arc uses 2000 for
the peak search.

| id | quantity | definition |
|---|---|---|
| A1 | Dirichlet exactness | `e_D = max(max|u(0,y)|/u_δ, max|u(L,y) − u_δ|/u_δ, max|v(x,0)|/max|v_ref|)` |
| A2a | free-edge residual, PINN on the boundary | RMS and max of `|P·N|/σ_n` on `gauge_top` and `right_arc`; value at the PINN's boundary peak |
| A2b | free-edge residual, like for like | on FEM boundary elements of the arc: `ρ_arc = RMS|P_PINN(c_e)·N_e| / RMS|P_FEM,e·N_e|`, where `c_e` is the element centroid and `N_e` the edge normal |
| A2c | other natural components | RMS `|P₂₁|/σ_n` on `x = 0` and on the grip face; RMS `|P₁₂|/σ_n` on `y = 0` |
| A3 | direct share of the peak | `δ_BC = vm/|σ_tt| − 1` at the PINN's boundary von Mises peak, in the deformed frame. `n = F⁻ᵀN/‖·‖`, `t = FT/‖·‖`; plane stress, so a traction-free point has `vm = |σ_tt|` |
| A4 | where the error lives | error-density ratio `DR_B = (share of ∫err² in band B) / (share of area in B)` for four bands of width `0.1 H_gauge`: free edge, grip face, `x = 0`, `y = 0`. Von Mises error at centroids, `v` error at nodes |
| A5 | end proximity | Spearman of the seed-mean in-range `|K_t err|` against `dx/H_grip`, over the 12 IR geometries |

### Predictions and decision rules

| | prediction | holds if | if it fails |
|---|---|---|---|
| **P-A1** | hard BCs are exact | `e_D ≤ 1e-6` on all 60 model × geometry pairs; falsified if any `> 1e-5` | a bug: fix before anything else |
| **P-A2** (rule R2) | the PINN is about as traction-free as the FEM on the arc | median `ρ_arc < 3` on every seed | natural BC is a distinguishing error source → C1–C3 |
| **P-A3** (rule R1) | the free-edge violation is **not** a main direct contributor to `K_t` error | mean IR `|δ_BC| < ⅓` of mean IR `|K_t err|` on at least 2 of 3 seeds | it is → C1–C3, starting with C2 |
| **P-A4** (rule R3) | no boundary layer at the Dirichlet grip face | median `DR_grip < 3` for both `v` and von Mises, on at least 2 of 3 seeds; the free-edge band has the largest von Mises `DR` | → C4–C5 |
| **P-A5** (rule R4) | end proximity does not drive error | Spearman `p ≥ 0.05` over the 12 IR geometries | → C4, and report `K_t` against `dx/H_grip` |

Expected overall outcome: P-A1 holds. P-A2 and P-A3 are the uncertain ones; I
expect both to hold, at roughly 55% and 65%. If all five hold, the boundary
conditions are **not** the explanation for the remaining error. That would be a
verified negative for the paper, and would point back to the interior
representation and optimiser (Phase 6.4).

---

## 5. Phase B — how much does the grip model matter? (FEM only)

- **B1 — clamp the grip face.** Re-solve all 20 geometries with `v = 0` added on
  `x = L_half`, at the same mesh (`h_factor` 1.3).
  - `K_t^fillet` = max von Mises over elements with `x ≤ x_g + dx/2` (this
    excludes the grip corner), divided by `σ_n`.
  - Also reported: `ΔN` and the relative `v` difference.

  **P-B1:** `|ΔK_t/K_t| ≥ 0.5%` on at least 4 of the 12 IR geometries. That is,
  the grip idealisation matters at a third or more of the operator's own `K_t`
  error, because the peak sits inside the end-effect zone (G1). I expect this
  at roughly 60%.
  - If it holds, the paper must state the grip idealisation, and B2 becomes
    necessary.
  - If it fails, `K_t` is robust to the grip choice at this level.
- **B2 (not yet implemented) — a prismatic grip section.**
  - Extend the geometry with a straight grip of length `ℓ ∈ {0.5, 1, 2} ×
    W_grip`, loaded rigidly.
  - Measure how `K_t` converges with `ℓ`, and compare against the Noda–Takase
    remote-tension value.
  - This needs a sixth boundary segment in `geometry/parametric_dogbone.py`,
    `geometry/triangulation.py` and the decoder's `x_max`, so it is a change to
    the problem, not a diagnostic. Estimated at a day, gated on P-B1.

---

## 6. Phase C — training experiments, gated on A and B

Each arm runs three seeds, 1600 epochs, trainer loop, bank 64, and is scored on
the 6.7 sets. Its prediction is committed before it runs.

| id | lever | gated on | why |
|---|---|---|---|
| C1 | more quadrature in a band along the free edge (per-element `n_per_elem` by band), or residual-driven adaptive sampling | R1 or R2 | quadrature at the concentration (Liu, Cai & Ramani 2023; Ye et al. 2026); already reinstated in 6.19 |
| C2 | `w_trac_arc` sweep, scored on `K_t` | R1 or R2 | wired but never scored on `K_t` (N3) |
| C3 | report the boundary stress from `φ` (mixed + energy) | R1 or R2 | exact tractions (Fuhg & Bouklas 2022); unstable in 6.7, so last |
| C4 | physics-based lift `g = u_δ S(x)/S(L)`, where `S(x) = ∫₀ˣ dξ/h(ξ)` is the 1D bar compliance | R3 or R4 | still exact and normalised; hands the network only the 2D correction |
| C5 | alternative normalised `u` multiplier near the grip | R3 | only if C4 is not enough |
| C6 | symmetry-exact inputs (`x²`, `y²` into the trunk), making both symmetry-plane shears exact | A2c symmetry residuals ≥ arc residual | turns two natural conditions into exact ones |
| C7 | soft Dirichlet control (penalty or AL on the grip face) | none: a reviewer control | the literature predicts worse (§2.1); run only for the paper's ablation table |

## 7. What this plan does not do

- Change the Dirichlet construction without evidence. It already meets the
  literature's criteria (D3).
- Repeat Phase 3's local-height `v` ansatz or its `w_trac_top` sweep. Both were
  measured.
- Adopt penalty, Nitsche or AL for the Dirichlet edges on the strength of the 3D
  comparisons. Those prefer AL only because distance functions are impractical
  there. Ours are three straight lines.

**Cost.** A: ~15 min CPU. B1: ~10 min. Each C arm: ~3 h (three seeds).

---


---

## 8. Results — Phase A and B1

```
python -m verification.bc_audit --report      # Phase A, from verification/results/bc_audit/audit.json
python -m verification.grip_sensitivity       # Phase B1, ~12 s
```

`verification/results/` is git-ignored, so the raw numbers behind every table
below are archived in `docs/`: `phase7_bc_audit.json`,
`phase7_grip_sensitivity.json`, and `phase6_7_retest_scores.json` (the `K_t`
errors A3 and A5 compare against).

### Verdicts

| | prediction | measured | verdict |
|---|---|---|---|
| **P-A1** | hard Dirichlet BCs are exact | max `e_D` = **0** over 60 model × geometry pairs | **HOLDS** |
| **P-A2** | the PINN is about as traction-free as the FEM on the arc (`ρ_arc < 3`) | median `ρ_arc` **3.45 / 3.29 / 3.43** | **FAILS**, narrowly |
| **P-A3** | the free-edge violation is not a main direct contributor to `K_t` error | mean `|δ_BC|` / mean `|K_t err|` = **0.94 / 0.25 / 0.56**, at least ⅓ on 2 of 3 seeds | **FAILS** (R1 fires) |
| **P-A4** | no boundary layer at the grip face; the free-edge band is worst for von Mises | `DR_grip` 1.2–1.4 (von Mises), 1.9–2.1 (`v`); `DR_free` for von Mises is the largest on 3 of 3 seeds | **HOLDS** |
| **P-A5** | end proximity does not drive `K_t` error | Spearman **−0.63**, p = 0.028 | **FAILS** (R4 fires), but confounded (below) |
| **P-B1** | clamping the grip moves `K_t` by at least 0.5% on at least 4 of 12 | **12 of 12**; mean 2.26%, max 4.16% | **HOLDS** |

Two of the five Phase A predictions held. Both of the uncertain ones (A2, A3)
failed in the same direction: the free edge matters more than I expected.

### A1 — the Dirichlet conditions are exact

`u(0, y)`, `u(L_half, y) − u_δ` and `v(x, 0)` are **exactly zero** on every
model and geometry. H1 is rejected.

This holds in the float32 arithmetic the model runs in. There, the float64 end
face rounds onto `x_max`, so the two grip checks are the same point. In float64
the end face and `x_max` differ by at most 4.9e-8 relative, which would give an
error of order 1e-8: still far below the 1e-6 threshold. Together with the audit (D1–D3), the Dirichlet side is correct,
consistent with the FEM, and built the way the literature recommends.

### A2 — the free edge is not traction-free enough

In-range medians over 12 geometries, in units of `σ_n`:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| arc, `|P·N|` RMS on the boundary | 2.48% | 2.22% | 2.33% |
| arc, max | 12.7% | 5.5% | 7.3% |
| at the boundary peak | 2.44% | 1.35% | 3.06% |
| gauge top, RMS | 0.93% | 1.12% | 1.18% |
| arc boundary elements, PINN | 2.54% | 2.28% | 2.43% |
| arc boundary elements, **FEM** | 0.73% | 0.73% | 0.73% |
| symmetry `x = 0` (`P₂₁`) / `y = 0` (`P₁₂`) / grip face (`P₂₁`) | 0.19 / 0.76 / 0.29% | 0.08 / 0.44 / 0.42% | 0.12 / 0.50 / 0.24% |

- **Free edge.** The network leaves about 2.3% of the nominal stress acting on
  the free arc. Measured at the same boundary-element centroids, that is 3.3–3.5×
  what the converged FEM leaves there through its own `O(h)` discretisation.
- **The ratio depends on the mesh; the absolute residual does not.** `ρ_arc`
  and P-A2's verdict are specific to the `h_factor` 1.3 reference: a finer FEM
  would shrink the denominator. The mesh-independent quantity is the network's
  own residual on the boundary, about 2.3% of `σ_n`.
- **Symmetry planes and grip face.** The natural conditions there are satisfied
  3–30× better than on the free edge, so C6 (symmetry-exact inputs) stays
  closed.

### A3 — at the peak, the violation pulls the prediction down

- **The normal stress at the peak is tensile.** `σ_nn/σ_tt` is +1.85%, +0.68%
  and +1.99% (median).
- **The shear is of similar size but barely matters.** Median `|σ_nt/σ_tt|` is
  0.94%, 0.68% and 1.05%. It enters von Mises only at second order (`3q²/2 ≈
  0.02%`), so it is the normal stress that moves the peak.
- **That lowers von Mises.** A tensile normal stress on top of tensile `σ_tt` is
  biaxial tension, so `vm < |σ_tt|`. Mean `δ_BC` is −0.79%, −0.44% and −0.88%.
- **The operator under-predicts `K_t` in range on every seed**: mean signed
  error −0.58%, −1.93% and −1.50%.
- **The violation's direct effect has the same sign as the bias.** Its size
  relative to each seed's mean bias is 1.37, 0.23 and 0.58 (0.52 pooled). On
  seed 0 it exceeds the whole bias.
- Per geometry, the two do not track each other consistently: Spearman(`δ_BC`,
  `K_t err`) = +0.39, −0.37, −0.24.

### A4 — no boundary layer at the Dirichlet face

Median error-density ratios, in range:

| band | von Mises s0 / s1 / s2 | `v` s0 / s1 / s2 |
|---|---|---|
| free edge | **2.57 / 2.46 / 2.31** | 1.77 / 2.01 / 1.98 |
| grip face | 1.35 / 1.17 / 1.26 | 2.05 / 1.92 / 2.03 |
| `x = 0` | 2.25 / 1.32 / 1.31 | 0.74 / 0.74 / 0.62 |
| `y = 0` | 0.69 / 0.74 / 0.76 | 0.02 / 0.02 / 0.02 |

- **von Mises.** The error is densest along the free edge, at about 2.5× its
  area share. No band reaches the 3× that would indicate a boundary layer.
- **`v`.** The error is spread about 2× near both the free edge and the grip
  face.

### A5 — the rule fired, but the test cannot tell grip proximity from a sharper concentration

Pre-registered as written, R4 fires: |`K_t` err| correlates with `dx/H_grip` at
−0.63 (p = 0.028, 12 geometries). But on this set `dx/H_grip` is almost a proxy
for the concentration itself:

| pair (12 in-range geometries) | Spearman | p |
|---|---|---|
| `dx/H_grip` vs reference `K_t` | **−0.93** | < 0.001 |
| \|`K_t` err\| vs reference `K_t` | +0.57 | 0.051 |
| \|`K_t` err\| vs `dx/H_grip`, controlling for reference `K_t` (partial, ranks) | −0.32 | 0.34 (df 9) |
| \|`K_t` err\| vs reference `K_t`, controlling for `dx/H_grip` | −0.04 | 0.90 (df 9) |

Geometries whose grip face sits close to the fillet also have the sharpest
concentrations. **Neither explanation survives controlling for the other**
(both partials null), so with n = 12 and the two variables correlated at
−0.93, this set cannot say which one drives the error.

The per-geometry numbers the rule asks to report (seed means, three energy
seeds):

| `dx/H_grip` | reference `K_t` | mean signed err | mean \|err\| |
|---|---|---|---|
| 0.76 | 1.295 | −2.61% | 2.61% |
| 0.89 | 1.244 | −2.62% | 2.62% |
| 0.92 | 1.232 | −1.67% | 1.67% |
| 0.96 | 1.196 | −1.86% | 1.86% |
| 0.99 | 1.231 | −1.67% | 1.67% |
| 1.05 | 1.204 | −2.63% | 2.63% |
| 1.07 | 1.178 | −0.03% | 0.68% |
| 1.11 | 1.199 | −0.27% | 0.68% |
| 1.12 | 1.175 | +0.38% | 0.68% |
| 1.19 | 1.177 | −0.63% | 0.63% |
| 1.21 | 1.170 | −2.05% | 2.05% |
| 1.21 | 1.171 | −0.38% | 0.38% |

The six geometries with `dx/H_grip ≤ 1.05` are each under-predicted by
1.7–2.6%. They include the five sharpest concentrations. Of the other six, five
are within 0.7% and one is at 2.05%.

### B1 — the grip idealisation moves `K_t` more than the network's error

Adding `v = 0` on the grip face changes the reference solution, all 20
geometries, same meshes:

| | in range (12) | out of range (8) |
|---|---|---|
| `ΔK_t/K_t` (fillet peak) | **−1.19% to −4.16%**, mean 2.26% | −4.60% to −4.89% |
| `ΔN/N` | +1.0% to +1.5% | +0.7% to +0.9% |
| `‖Δv‖/‖v‖` | 0.60–0.74 | 0.51–0.60 |
| global peak moves to the corner? | never | never |

- **Direction.** Clamping always lowers `K_t`.
- **Size.** In range, the shift tracks the sharpness of the concentration
  (Spearman +0.71 with reference `K_t`, p = 0.010) more than grip proximity
  (−0.48 with `dx/H_grip`, p = 0.12). With the two correlated at −0.93, the A5
  confound applies here too. A Saint-Venant reading (a closer grip gives a
  larger shift) is physically plausible but not established by this set.
- **Against the operator's error.** The in-range *mean* shift (2.26%) is larger
  than the operator's own mean `K_t` error (1.51%). The smallest shift (1.19%)
  is below it.
- **`v`.** The transverse displacement is **mostly a property of the grip
  model**: 60–74% of it changes. That is not a PINN-vs-FEM error, because both
  solvers share the grip. It is a caveat on what the `v` numbers mean.

### Exploratory — post hoc, hypotheses rather than results

Phase 6.21 is why these are labelled this way. None of them was predicted.

- **E1 — the direct gain from a traction-free edge is estimated, to first order,
  at 12–22%.**
  - Method: divide `δ_BC` out of the predicted peak. This first-order estimate
    is what `K_t` would read if the edge were traction-free and `σ_tt`
    unchanged.
  - Result: mean |`K_t` err| goes 0.97 → 0.80%, 1.93 → 1.70% and 1.64 → 1.28%.
    That is 12–22% lower, better on 6, 8 and 8 of 12 geometries.
  - This mixes locations: `δ_BC` from the boundary peak is applied to a `K_t`
    taken over element centroids. It is an estimate, not a bound.
  - After the correction, seeds 1 and 2 are still too low (−1.49%, −0.63%): the
    remainder there is `σ_tt` itself. On seed 0 the correction overshoots to
    +0.22%.
  - Any Phase C lever that only repairs the edge should be predicted at about
    this size. That is below what three seeds resolve here: seed-to-seed CV of
    mean |`K_t` err| is about 33% in the trainer loop.
- **E2 — the free-edge residual separates in-range from out-of-range geometries
  on every seed, with no overlap.**
  - Arc `|P·N|` RMS is 1.7–5.0% in range and 9.2–27.7% out of range.
  - It needs no reference. The run above normalised by the FEM's `σ_n`, so the
    check was repeated with the network's own gauge-mean stress (a uniform grid,
    no FEM). The ranges are identical to one decimal, still with no overlap on
    any seed. The network's nominal agrees with `σ_n` to within −0.2..+0.6%.
  - But these out-of-range geometries leave the range in one direction only
    (taper). The flag also tracks "out of range", not large error: the
    out-of-range `K_t` errors were +1.9%, +1.7% and +7.2%.
  - It is a candidate **reference-free out-of-range flag**. It needs a
    pre-registered test on geometries out of range in other parameters.
  - Phase 6.7 killed the constitutive-gap version of this claim in exactly that
    kind of test.
- **E3 — the transverse field is grip-dominated** (B1, `‖Δv‖/‖v‖` 0.51–0.74).
  Phase 3–6's long effort on `v` was spent on a quantity whose physical meaning
  depends mostly on the grip idealisation.

### Errata to §1 (found after the run by an independent check)

§1 was written before the run and is left as committed. These corrections
apply to it:

- **N2, peak location.** On the 6.7 sets the reference peak spans `x*/x_g` =
  1.03–1.12, not 1.03–1.11. The latter was the older eight-geometry range.
- **N2, "never measured".** Phase 3 logged the *gauge-top* traction loss for
  energy-form runs, as a loss value. No free-edge residual on the **arc** had
  been measured.
- **N3, scope.** The Phase 3 sweep also scored the section-force error, not only
  `v` and L2.
- **D2, `x_max` and `y_max`.** They are maxima over all of the geometry's nodes,
  boundary and interior. The boundary endpoints attain them, so the values are
  as stated.
- **D5, normalisation.** The normal derivative of `1 − y/h(x)` on the arc is
  `√(1 + h′²)/h(x)`. On the default geometry it stays at 0.83–1.0× of its
  gauge-top value along the arc, because the `1/h` factor offsets the slope.
  The "about 1.7" figure was the slope factor alone.
- **G2, the clamped-corner singularity.**
  - Williams' clamped–free eigen-equation crosses `λ = 1` at
    `sin²α = (κ + 1)/4`.
  - For this material (ν ≈ 0.23, plane stress, κ = 2.25) that is **64.4°**.
  - So clamping would make only 2 of the 8 out-of-range corners (64.8°, 67.0°)
    weakly singular, not all of them.
  - Consistently, B1 never saw the peak move to the corner.

### What the gates now say

- **R1 and R2 fired, so C1–C3 are open, C2 first** (`w_trac_arc`, scored on
  `K_t`).
  - Its prediction must carry E1's bound: about a 20% reduction in |`K_t` err| if
    only the direct effect is recovered.
  - It needs five seeds or a paired-seed design; three cannot resolve that.
  - A larger gain would mean the edge also drives `σ_tt`, which is the
    interesting outcome.
- **R4 fired, so C4 is open.** Its prediction must be written against both
  readings of A5:
  - if grip proximity drives the error, a physics-based lift should help most
    where `dx/H_grip` is small;
  - if the sharpness of the concentration drives it, the lift should not help.
- **P-B1 held, so B2 is no longer optional** if the paper's `K_t` is to be the
  `K_t` of a standard specimen.
  - It changes the problem family: a new geometry, new references, and
    retraining.
  - That is the authors' decision, not a diagnostic.

### The answer to the question that started this phase

**Are the Dirichlet conditions correct?**
- Yes. They are exact (zero error on every model and geometry), identical to
  the FEM's, and built the way the 2022–2026 literature recommends.
- They are not a source of the error against FEM.

**Do boundary conditions matter?** Yes, in two ways that have nothing to do
with how the Dirichlet conditions are implemented:

1. **The free edge.** The energy form leaves the curved edge about 3× less
   traction-free than the FEM does.
   - At the peak this reads the stress about 0.4–0.9% low, in the same direction
     as the operator's average under-prediction.
   - Removing it would shrink the `K_t` error by an estimated 12–22%. This is a
     first-order, exploratory estimate.
2. **The grip model.** Which Dirichlet condition is placed on the grip face is a
   modelling choice.
   - Frictionless and clamped differ by 1.2–4.2% in `K_t` in range (mean 2.3%,
     above the network's mean error of 1.5%) and by about 4.7% out of range.
   - The grip face sits about one half-width from the peak.
   - The paper has to state this choice. To speak about standard specimens it
     needs a grip section (B2).

## References

Checked against publisher or arXiv pages.

- Basir, S., Senocak, I. (2022). Physics and equality constrained artificial neural networks. *J. Comput. Phys.* 463:111301. doi:10.1016/j.jcp.2022.111301
- Berg, J., Nyström, K. (2018). A unified deep artificial neural network approach to partial differential equations in complex geometries. *Neurocomputing* 317:28–41. doi:10.1016/j.neucom.2018.06.056
- Berrone, S., Canuto, C., Pintore, M., Sukumar, N. (2023). Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks. *Heliyon* 9(8):e18820. doi:10.1016/j.heliyon.2023.e18820
- Dean, A., Bahtiri, B. (2026). PI-GINOT: Data-free geometry-informed neural operator learning for finite-strain hyperelasticity on parametric DogBone specimens. arXiv:2607.23299
- E, W., Yu, B. (2018). The Deep Ritz Method. *Commun. Math. Stat.* 6(1):1–12. doi:10.1007/s40304-018-0127-z
- Fuhg, J.N., Bouklas, N. (2022). The mixed Deep Energy Method for resolving concentration features in finite strain hyperelasticity. *J. Comput. Phys.* 451:110839. doi:10.1016/j.jcp.2021.110839
- Garrell, M.G., Shih, A.J., Lara-Curzio, E., Scattergood, R.O. (2003). Finite-element analysis of stress concentration in ASTM D 638 tension specimens. *J. Test. Eval.* 31(1):52–57. doi:10.1520/JTE12359J
- Göschel, N., Götschel, S., Ruprecht, D. (2025). Enforcing boundary conditions for physics-informed neural operators. arXiv:2510.24557
- Horgan, C.O. (1989). Recent developments concerning Saint-Venant's principle: an update. *Appl. Mech. Rev.* 42(11):295–303. doi:10.1115/1.3152414
- Huang, J., Wang, H., Zhou, T. (2022). An augmented Lagrangian deep learning method for variational problems with essential boundary conditions. *Commun. Comput. Phys.* 31(3):966–986. doi:10.4208/cicp.OA-2021-0176
- Kardak, A.A., Bilich, L.A., Sinclair, G.B. (2017). Stress concentration factors for ASTM E8/E8M-15a plate-type specimens for tension testing. *J. Test. Eval.* 45(6):2294–2298. doi:10.1520/JTE20160385
- Lagaris, I.E., Likas, A., Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Trans. Neural Netw.* 9(5):987–1000. doi:10.1109/72.712178
- Li, Z. et al. (2023). Geometry-informed neural operator for large-scale 3D PDEs. *NeurIPS 36*. arXiv:2309.00583
- Liao, Y., Ming, P. (2021). Deep Nitsche Method: Deep Ritz method with essential boundary conditions. *Commun. Comput. Phys.* 29(5):1365–1384. doi:10.4208/cicp.OA-2020-0219
- Liu, M., Cai, Z., Ramani, K. (2023). Deep Ritz method with adaptive quadrature for linear elasticity. *Comput. Methods Appl. Mech. Eng.* 415:116229. doi:10.1016/j.cma.2023.116229
- Liu, Q., Zhong, W., Meidani, H., Abueidda, D., Koric, S., Geubelle, P. (2026). Geometry-informed neural operator transformer for partial differential equations on arbitrary geometries. *Comput. Methods Appl. Mech. Eng.* 451:118668. doi:10.1016/j.cma.2025.118668
- Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F., Johnson, S.G. (2021). Physics-informed neural networks with hard constraints for inverse design. *SIAM J. Sci. Comput.* 43(6):B1105–B1132. doi:10.1137/21M1397908
- McClenny, L.D., Braga-Neto, U.M. (2023). Self-adaptive physics-informed neural networks. *J. Comput. Phys.* 474:111722. doi:10.1016/j.jcp.2022.111722
- Müller, J., Zeinhofer, M. (2022a). Error estimates for the deep Ritz method with boundary penalty. *PMLR* 190:215–230 (MSML). arXiv:2103.01007
- Müller, J., Zeinhofer, M. (2022b). Notes on exact boundary values in residual minimisation. *PMLR* 190:231–240 (MSML). arXiv:2105.02550
- Noda, N.-A., Takase, Y. (2004). Stress concentration factor formulas useful for any dimensions of shoulder fillet in a flat test specimen under tension and bending. *J. Test. Eval.* 32(3):217–226. doi:10.1520/JTE11799
- Oldenburg, J. et al. (2022). Geometry aware physics informed neural network surrogate for solving Navier–Stokes equation (GAPINN). *Adv. Model. Simul. Eng. Sci.* 9:8. doi:10.1186/s40323-022-00221-z
- Ouyang, Y., Jiang, W., Du, W., Zhang, J., Chen, Y., Zheng, H. (2026). A deep energy method for solid mechanics based on a generalized multiplier approach. *Eng. Anal. Bound. Elem.* 183:106606. doi:10.1016/j.enganabound.2025.106606
- Rao, C., Sun, H., Liu, Y. (2021). Physics-informed deep learning for computational elastodynamics without labeled data. *J. Eng. Mech.* 147(8):04021043. doi:10.1061/(ASCE)EM.1943-7889.0001947
- Rowan, C., Maute, K., Hampleman, K., Doostan, A. (2025). Boundary condition enforcement with PINNs: a comparative study and verification on 3D geometries. arXiv:2512.14941
- Saad, N., Gupta, G., Alizadeh, S., Maddix, D.C. (2023). Guiding continuous operator learning through physics-based boundary constraints (BOON). *ICLR 2023*. arXiv:2212.07477
- Son, H., Cho, S.W., Hwang, H.J. (2023). Enhanced physics-informed neural networks with augmented Lagrangian relaxation method (AL-PINNs). *Neurocomputing* 548:126424. doi:10.1016/j.neucom.2023.126424
- Sukumar, N., Roy, A. (2026). A Wachspress-based transfinite formulation for exactly enforcing Dirichlet boundary conditions on convex polygonal domains in physics-informed neural networks. *Comput. Mech.* 78(3):1015–1044. doi:10.1007/s00466-026-02789-4
- Sukumar, N., Srivastava, A. (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. *Comput. Methods Appl. Mech. Eng.* 389:114333. doi:10.1016/j.cma.2021.114333
- Wang, S., Teng, Y., Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5):A3055–A3081. doi:10.1137/20M1318043
- Wang, S., Wang, H., Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Sci. Adv.* 7:eabi8605. doi:10.1126/sciadv.abi8605
- Wang, S., Yu, X., Perdikaris, P. (2022). When and why PINNs fail to train: a neural tangent kernel perspective. *J. Comput. Phys.* 449:110768. doi:10.1016/j.jcp.2021.110768
- Wang, Y., Sun, J., Rabczuk, T., Liu, Y. (2024). DCEM: A deep complementary energy method for linear elasticity. *Int. J. Numer. Methods Eng.* 125(24):e7585. doi:10.1002/nme.7585
- Williams, M.L. (1952). Stress singularities resulting from various boundary conditions in angular corners of plates in extension. *J. Appl. Mech.* 19(4):526–528.
- Xie et al. (2025). Spectral analysis of hard-constraint PINNs: the spatial modulation mechanism of boundary functions. arXiv:2512.23295
- Ye, Y., An, D., Li, H., Wang, W., Li, X., Shi, P. (2026). High-precision stress prediction using deep energy network enhanced by stress equilibrium with Delaunay integration on refined grid. *Cell Rep. Methods* 6(6):101424. doi:10.1016/j.crmeth.2026.101424
- Zeinhofer, M., Masri, R., Mardal, K.-A. (2025). A unified framework for the error analysis of physics-informed neural networks. *IMA J. Numer. Anal.* 45(5):2988–3025. doi:10.1093/imanum/drae081
- Zhong, W., Meidani, H. (2025). Physics-informed geometry-aware neural operator. *Comput. Methods Appl. Mech. Eng.* 434:117540. doi:10.1016/j.cma.2024.117540
- Zhu, Wang, Zhang, Rabczuk (2026). WINO: a weak-form physics-informed neural operator for hyperelasticity on variable domains. arXiv:2605.24651
