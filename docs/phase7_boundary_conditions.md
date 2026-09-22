# Phase 7 — boundary conditions: are they right, and are they the error?

**Status.** Audit and literature review complete. Phase A (evaluation-only) and
Phase B1 (FEM-only) are specified below with their predictions and decision
rules, **committed before either is run**. Phase C (training) is proposed and
gated on A and B; nothing in it has been run.

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
