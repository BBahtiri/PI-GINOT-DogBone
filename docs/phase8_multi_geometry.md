# Phase 8 — beyond the dog-bone: distance-function Dirichlet conditions and four new specimen families, validated against FEM

**Status.**
- **8.1 (the distance-function Dirichlet layer) is done and verified.** An
  independent check then found three numerical defects. They are fixed and
  now regression-tested (`cee6e49`, §3).
- The same check found several overreaching statements in the first version
  of this plan. They are corrected in place and listed in §10.
- **8.2 has started.** The FEM reference was run on the first new family (the
  open-hole plate) and checked against the classical solution.
- Everything else below is the plan.
- Predictions for 8.1 and 8.2 were committed with their scripts before those
  ran (`fbc41f4`, `9a402a4`). Predictions for later steps are written here and
  get their own commit before each run.

---

## In plain words

**Do we validate against FEM?**
- Yes. Every number in this project is scored against an independent
  finite-element solution. That solution is itself checked against exact
  answers before it is trusted.
- But until now only on dog-bones. The geometry generator, the meshes, the test
  sets and the metrics were all dog-bone-specific.

**What this phase adds.**
- Four new specimen families, each chosen to test something the dog-bone
  cannot:
  - a plate with a hole: new topology, the textbook stress concentration;
  - a plate with two edge notches: sharper peaks;
  - a plate with a bonded rigid inclusion: the displacement is prescribed on a
    *curved* edge, which is what distance functions are for;
  - a plate with one edge notch, clamped at the grips: no symmetry, and bending.
- A general way of imposing the fixed-displacement boundary conditions
  (distance functions). It turns out to contain today's dog-bone formula as an
  exact special case.

**What is already done.**
- The distance-function layer passes all its exactness checks on six
  geometries.
- It reproduces the current network's boundary formula to machine precision.
- Where two fixed-displacement edges meet at a corner, the two standard ways
  of building the formula behave differently. Which one gives better stresses
  is an open question; the plan tests it rather than assuming an answer.
- The FEM solver, unchanged, reproduces the handbook stress concentration of a
  plate with a hole to within 1%.
- That plate is also insensitive to how its grip is modelled (0.006%; 0.03–0.06%
  for the shortest plates in range), unlike the dog-bone (1–4%).

---

## 1. Where validation stands

- **The reference.** Every result since Phase 1 is scored against
  `verification/fem/solver.py`, verified on the dog-bone before anything is
  quoted:
  - constant-strain triangles, Newton with line search, the same Neo-Hookean
    plane-stress law;
  - patch test 1.5e-16;
  - uniaxial closed form;
  - mesh convergence: `K_t` bias 0.19% at `h_factor` 1.3.
- **The gap.** Only dog-bones have ever been solved.
  - About 50 modules import dog-bone-specific geometry (`get_fillet_geometry`,
    `x_g`, `R_fillet`, `W_gauge`).
  - `generate_dogbone` rejects holes outright.
  - The mesher, the collocation sampler, the held-out sets, the metrics and the
    oracle conditioning all assume the dog-bone's five segments.

A paper that claims a *geometry-informed* operator needs more than one geometry
family. That is the purpose of this phase.

## 2. The families

All are plane-stress, compressible Neo-Hookean, and loaded in x-tension by a
prescribed grip displacement, exactly as the dog-bone.

| | family | model | Dirichlet boundary | where the peak is | classical small-load reference | what it tests |
|---|---|---|---|---|---|---|
| F1 | dog-bone (existing) | quarter | `u = 0` on `x = 0`, `u = u_δ` on `x = L`, `v = 0` on `y = 0` | free fillet arc | Noda & Takase (2004) | baseline |
| **F2** | **open-hole plate** | quarter | as F1 | free hole edge (crown) | Peterson (Howland 1930), Heywood (1952) | new topology; `K_t` 2.1–2.6 |
| **F3** | **double-edge U-notched plate** | quarter | as F1 | free notch root | Peterson Ch. 2; Noda, Sera & Takase (1995) | sharper gradients; `K_t` ≈ 2–4 |
| **F4** | **plate with bonded rigid inclusion** | quarter | as F1, plus `u = v = 0` on the inclusion arc | **on the Dirichlet arc** | Goodier (1933), derived symbolically at step 8.2 | curved Dirichlet boundary; corners where Dirichlet pieces meet |
| **F5** | **single-edge U-notched plate, clamped grips** | half (no `y`-symmetry) | `u = 0` on `x = 0`; grip clamped, `u = u_δ` and `v = 0` | free notch root | Peterson Ch. 2 (single notch) | no symmetry plane for `v`; tension–bending coupling; clamped-vs-pinned sensitivity known from SENT testing |

**Design rule carried over from Phase 7 B1.** Every new family keeps its grips
at `L_half ≥ 1.5 W`, so the reference `K_t` does not depend on how the grip is
idealised.
- Verified for F2 (§4): clamping moves `K_t` by 0.006% at `L_half/W = 2`, and by
  0.03–0.06% at `L_half/W = 1.5` for `d/W` 0.2–0.6.
- The dog-bone moves by 1.2–4.2%.
- F5 is the deliberate exception: clamping is part of its definition.

**Parameter ranges (training bank; the held-out sets use a 12 in-range + 8
out-of-range split per family, as in 6.7).**

| family | parameters | in-range | out-of-range direction |
|---|---|---|---|
| F2 | `W`, `d/W`, `L_half/W` | `W` 16–26, `d/W` 0.2–0.5, `L_half/W` 1.5–2.5 | `d/W` 0.55–0.6 |
| F3 | `W`, notch depth `t/W`, root radius `ρ/t`, `L_half/W` | `t/W` 0.1–0.3, `ρ/t` 0.25–1 | `ρ/t` 0.15–0.25 (sharper) |
| F4 | as F2 | as F2 | `d/W` 0.55–0.6 |
| F5 | as F3, `L_half/W` 2–3 | as F3 | `t/W` 0.3–0.35 |

**Tiers.**
- **F2 and F3** are the core of the concentration-factor validation.
- **F4** is where the distance-function layer is necessary: a curved Dirichlet
  arc that meets straight pieces. So is Phase 7's clamped grip *segment* (B2).
- **F5** still has only full, parallel Dirichlet edges, so closed forms would
  do. What it changes is the configuration: no symmetry plane for `v`, and a
  clamped grip.
- If time forces a cut, keep F2 and F4: one new topology and one curved
  Dirichlet boundary.

## 3. The distance-function Dirichlet layer (8.1, done)

`geometry/adf.py` implements Sukumar & Srivastava (2022) in the form Berrone et
al. (2023) found most accurate.
- **Per piece:** first-order normalised distance functions, one per Dirichlet
  piece. Primitives are lines, trimmed segments and circles.
- **Joining pieces:** R-equivalence (`m = 1`).
- **The data:** a transfinite-interpolation lift.
- **Numerics:** both are evaluated in product form, so nothing divides by a
  vanishing distance.
- **Specs:** for F1–F4, for F5, and for Phase 7's clamped grip *segment* (B2).

**The dog-bone ansatz is the special case.**
- `R(x, L − x) = x(L − x)/L`, and the transfinite weight of the grip is `x/L`.
- So `u = u_δ ξ + ξ(1 − ξ) N_u` and `v = (y/H) N_v`: what the decoder has always
  done, identical in exact arithmetic.
- In floating point the two agree to rounding (9.3e-17 in float64, 2.5e-8 in
  float32), not bit for bit. A decoder regression test must use a tolerance,
  or keep the legacy path for the dog-bone.
- For two *parallel* pieces the normalised form and a plain product differ
  only by a constant. So for the dog-bone the distinction is moot; it matters
  where pieces meet or are not parallel.

**Level-0 verification** (`python -m verification.adf_verify`, float64,
thresholds fixed in `fbc41f4` before the run; output in
`docs/phase8_adf_verify.txt`):

| case | zero set | `|∂φ/∂n − 1|` | interior min φ | lift error | ansatz error |
|---|---|---|---|---|---|
| dog-bone | 3.6e-15 | 7.4e-11 | > 0 | 2.2e-16 | 2.2e-16 |
| open hole | 7.1e-15 | 5.0e-11 | > 0 | 2.2e-16 | 1.1e-16 |
| double notch | 7.1e-15 | 5.0e-11 | > 0 | 2.2e-16 | 1.1e-16 |
| rigid inclusion | 7.1e-15 | 6.5e-9 | > 0 | 2.2e-16 | 1.1e-16 |
| single notch, clamped | 7.1e-15 | 5.0e-11 | > 0 | 2.2e-16 | 1.4e-16 |
| clamped grip segment | 1.8e-15 | 2.0e-10 | > 0 | 1.1e-16 | 1.1e-16 |

- **Against the shipped decoder** (`PhysicsDecoder._apply_hard_bc`): maximum
  relative difference **9.3e-17**. All checks pass.
- **Three defects found by the independent check, fixed in `cee6e49`,** each now
  a regression check that fails on the pre-fix code:
  - float32 NaN at corner nodes: the division guard underflowed;
  - NaN gradients exactly on a trimmed segment and at its ends;
  - a corner lift that returned 0 where two pieces share non-zero data.
- **What differs where Dirichlet pieces meet.** On the inclusion arc,
  approaching the corner `(0, r)` where it meets the symmetry line:

| `∂φ_u/∂n`, relative to mid-arc | `s = 0.1 r` | `0.01 r` | `0.001 r` |
|---|---|---|---|
| R-equivalence | 1.0000 | 1.0000 | 1.0000 |
| plain product of the distances | 0.150 | 0.015 | 0.0015 |

- **This is a property, not a verdict.**
  - R-equivalence keeps unit normal derivative up to the corner. Its Laplacian
    is unbounded at the vertex (Berrone et al., App. A.1), which is harmless to
    the energy form: it uses first derivatives only.
  - The product's normal derivative decays linearly. But where two zero-data
    pieces meet at 90°, as at both of F4's corners, the exact solution's
    gradient vanishes at the corner too. So the product's decay has the right
    shape there, and R-equivalence asks the network to learn `N → 0`.
  - F4's peak (`σ_rr` at `(r, 0)`) is set by `u`'s multiplier, which has no
    corner there.
  - So no stress difference is predicted at the peak. Which construction gives
    the better field near the corners is ablation A-ADF (§6).
  - The earlier reading, that the product "loses control of the strain", was
    an overreach. Berrone et al.'s reported failure of the product is about
    values that become excessively small near multiple segments, not this
    corner behaviour.

## 4. The FEM reference on a new family (8.2, started)

`geometry/shapes.py` is a shape-generic mesher.
- It uses the dog-bone mesher's recipe against a small `Shape` interface. The
  verified dog-bone mesher is untouched.
- The first new family is the quarter open-hole plate.
- Segment names follow the solver's Dirichlet convention, so
  `verification/fem/solver.py` runs **unchanged**.

`python -m verification.open_hole_fem` (predictions fixed in `9a402a4`; raw
numbers in `docs/phase8_open_hole_fem.json`). Setup: plate `L_half = 40`,
`W = 20`, four meshes per `d/W` at 30–85 elements per hole radius, and
`K_tn = max P₁₁ / (F/(H − r))` at small load (strain 2.5e-5):

| `d/W` | `K_tn`, four meshes | extrapolated | Peterson | Heywood | vs Peterson |
|---|---|---|---|---|---|
| 0.2 | 2.504 / 2.509 / 2.510 / 2.508 | 2.507 | 2.506 | 2.512 | +0.01% |
| 0.3 | 2.339 / 2.360 / 2.357 / 2.358 | 2.354 | 2.347 | 2.343 | +0.32% |
| 0.4 | 2.242 / 2.245 / 2.241 / 2.244 | 2.241 | 2.233 | 2.216 | +0.37% |
| 0.5 | 2.179 / 2.164 / 2.169 / 2.170 | 2.176 | 2.156 | 2.125 | +0.95% |

| | prediction | measured | verdict |
|---|---|---|---|
| P-8.2a | extrapolated `K_tn` within 1.5% of Peterson at every `d/W` | +0.01 to +0.95% | **HOLDS** |
| P-8.2b | finest mesh within 1% of the extrapolation | 0.04–0.31% | **HOLDS** |
| P-8.2c | clamping the grip moves `K_tn` by < 0.3% | **0.006%** | **HOLDS** |
| P-8.2d | the 1 mm load moves `K_tn` by < 3% | **−5.69%** | **FAILS** |

**What P-8.2d says.**
- At the project's load (nominal strain 2.5%; 8.2% at the crown element), the
  finite-strain `K_t` sits well below the linear handbook value.
- Exploratory split: with the dog-bone's definition (peak Cauchy von Mises over
  a far-field mean) the drop is −3.1% (3.731 → 3.614). So about half of the
  5.7% is the Piola-versus-Cauchy definition, and about half is genuine
  finite-strain behaviour.
- Consequences:
  - the handbook formulas are **small-load checks only**;
  - the finite-strain FEM is the reference;
  - `K_t` must be defined the same way in every family (§5).

**What the table does not say.**
- **The mesh sequence is not monotonic.** The spread across the four meshes is
  0.15–0.89%.
- **So "extrapolated" is a least-squares line through noise**, not a Richardson
  limit.
- **The cause is mostly the meshes being random and non-nested**, and partly
  the peak extraction. The independent check found:
  - at a fixed density, the peak moves 0.56–0.72% across mesher seeds at 42
    elements per radius, and 0.11–0.13% at 85;
  - boundary-extrapolated estimators did not shrink the four-mesh spread (0.26%
    to 2.9%, depending on the estimator).
- **Why it matters, and how much.**
  - At the finest level the noise is about 0.1%, below the operator errors to
    be measured (~1–2%). The coarse meshes are the problem.
  - The fix for step 8.2b is nested or structured crown meshes (or averaging
    over mesher seeds), not a cleverer estimator.

## 5. What has to be generalised

| component | today | needed | effort |
|---|---|---|---|
| geometry | `generate_dogbone`, 5 fixed segments | `Shape` classes F2–F5 (F2 done), boundary tags, point-in-domain | 1–2 days |
| FEM mesher | dog-bone only | `build_shape_mesh` (done), plus robust peak extraction | ½ day |
| FEM solver | generic, keyed on segment names | unchanged | — |
| Dirichlet layer | `_apply_hard_bc` (dog-bone) | `DirichletSpec` per family in the decoder. Regression test: dog-bone output bit-identical (8.1 shows the math is) | ½ day |
| energy quadrature | `cached_mesh(params)` | per-shape cached meshes, same stratified resampling | ½ day |
| collocation / point cloud | 5 segments | per-shape boundary sampling (encoder input) | 1 day |
| oracle conditioning | 4 dog-bone parameters | per-family parameter vector; family one-hot for a joint model | ½ day |
| banks, held-out sets | dog-bone ranges | per family, IR/OOR split, frozen seeds | ½ day |
| metrics | `K_t` = peak vm / gauge-mean vm | one definition for all families: peak Cauchy vm over the mean Cauchy vm of a family-specific nominal region (gauge; far-field band; net ligament) | ½ day |
| free-edge residual (Phase 7 A2) | arc + gauge top | every free boundary per shape | ½ day |

## 6. Protocol for the paper

**Tier 1 (must-have): one operator per family.**
- Same recipe as the 6.7 re-test: energy form, oracle-conditioned, trainer loop,
  bank 64, 1600 epochs, three seeds (five where a comparison is to be made).
- Each operator is scored on its family's 12 + 8 held-out geometries against
  the finite-strain FEM.
- Metrics:
  - displacement and von Mises L2;
  - `K_t` R² and |error| against the predict-the-mean baseline;
  - peak location and section force;
  - the free-edge residual.
- F2–F4 are designed to be grip-insensitive, so their `K_t` means what the
  handbook's `K_t` means (in the small-load limit). F5 is clamped by
  definition.

**Tier 2 (the geometry-informed claim): one operator across families.**
- An oracle version (family one-hot plus padded parameters) and a point-cloud
  version.
- Phase 4 found the point-cloud encoder weak even within one family. Tier 2
  will show whether that holds across families, which is the question a GINOT
  paper has to answer.

**Two ablations that only these families make possible.**
- **A-ADF (F4).** R-equivalence against the plain product in the Dirichlet
  layer.
  - No winner is predicted (§3).
  - The comparison is of field errors near the arc's two corners. A `K_t`
    difference is not expected, because the peak's multiplier has no corner.
- **F2 vs F4, descriptive only.**
  - The first version of this plan pre-registered "F4's in-range `K_t` error is
    below F2's on every seed". **It is withdrawn, before any run, as
    confounded.**
  - F4's concentration is lower and smoother: Goodier's solution gives
    `σ_rr,max ≈ 1.5σ` for this material, against about `3σ` at a hole. A pass
    would not isolate the free-versus-Dirichlet mechanism.
  - Its stated reason was also misleading: the construction makes the
    *displacement* exact on the arc, not the normal strain that sets the peak.
  - Replacement: scale-free metrics (R², |error|/`K_t`), reported side by side
    with no mechanistic claim.
  - The mechanism test stays where Phase 7 put it: exact tractions on the same
    family (C3), against the natural condition.

**Verification per family, before any scoring** (the dog-bone's hierarchy,
repeated):
- Level 0: the Dirichlet layer (§3, done for all five).
- Level 3: FEM against the classical small-load `K_t`:
  - F2 done;
  - F3 and F5 against Peterson/Noda;
  - F4 against Goodier's solution, derived with sympy. It is an infinite-plate
    solution, so the check uses a wide plate (`d/W ≤ 0.1`); the finite-width
    references then rest on the verified mesher and solver.
- Mesh convergence with the robust peak extraction.
- A B1-style grip check.

## 7. Next steps, in order

1. **8.2b — convergent references.** Nested or structured crown meshes (or
   averaging over mesher seeds).
   - Prediction: monotonic convergence, and a seed-to-seed spread < 0.1% at the
     finest level.
2. **8.2c — `Shape` classes F3, F4, F5**, each with Level-3 and grip checks
   (predictions per family, committed first). F4 needs Goodier's `K_t` derived
   symbolically.
3. **8.3 — plumbing.**
   - `DirichletSpec` in the decoder, with the dog-bone bit-identical regression.
   - Per-shape quadrature, collocation, point clouds and banks.
   - Unified `K_t`.
4. **8.4 — Tier 1 training.** Four families × three seeds.
   - About 1 h per seed on this CPU, much less on a GPU.
   - Predictions per family committed first, including A-free vs Dirichlet.
5. **8.5 — Tier 2** and the two ablations.
6. **8.6 — paper assets.**
   - Per-family `K_t` parity plots.
   - A families × metrics table.
   - The verification appendix: Level 0 for the Dirichlet layer, Level 3 per
     family.

## 8. Risks

- **Notch roots (F3, F5).** Steep gradients need a finer quadrature mesh at the
  root. The energy form's free-edge weakness (Phase 7) is expected to be
  largest there.
- **F5's clamped grips.** They make 90° clamped–free corners. That is above the
  64.4° threshold for this material (Williams 1952, Phase 7 errata), so the
  corners are weakly singular. Keep the grips `≥ 2W` from the notch and search
  for `K_t` only near the notch, as B1 did.
- **Handbook formulas are linear.** At the project's load, finite-strain `K_t`
  departs by 3–6% (P-8.2d). They check the reference at small load, never the
  operator at full load.
- **Tier 2 may fail.** A single operator across families could simply not work
  with the current encoders. That would be a result, not a reason to drop Tier
  1.

## 9. Decisions for the authors

1. Confirm the four families and the parameter ranges in §2.
2. The paper's main claim: per-family operators (safe), or one cross-family
   operator (the stronger geometry-informed claim, uncertain).
3. The dog-bone's grip. The new families are grip-insensitive by design, so the
   dog-bone's 1–4% grip sensitivity (Phase 7 B1) will stand out. Adding a
   prismatic grip section (Phase 7 B2) would put all five families on the same
   footing.
4. Compute. Tier 1 on four families is about 12–20 CPU-hours at three seeds. A
   GPU run would change the seed count we can afford.

## 10. Corrections to the first version of this plan

An independent check of every number and claim was run after `53cc47c`.
Corrected above:
- d/W 0.2 table entries: 2.509 and 2.506.
- Crown strain: 8.2%, not "about 9%".
- The cause of the non-monotonic convergence (mostly the meshes).
- The corner interpretation in §3: a property, not the failure Berrone et al.
  report.
- "Already the normalised construction, which is why it passed the audit":
  moot for parallel pieces.
- "Bit-identical" → equal to rounding.
- The grip rule: `L_half ≥ 1.5 W`, measured.
- F5 does not need the general layer.
- "F2–F5 grip-insensitive" → F2–F4.
- The confounded F4-vs-F2 prediction, withdrawn before any run.

## References (added in this phase)

- Goodier, J.N. (1933). Concentration of stress around spherical and cylindrical inclusions and flaws. *J. Appl. Mech.* 1(2):39–44.
- Heywood, R.B. (1952). *Designing by Photoelasticity*. Chapman & Hall.
- Howland, R.C.J. (1930). On the stresses in the neighbourhood of a circular hole in a strip under tension. *Phil. Trans. R. Soc. A* 229:49–86.
- Noda, N.-A., Sera, M., Takase, Y. (1995). Stress concentration factors for round and flat test specimens with notches. *Int. J. Fatigue* 17(3):163–178.
- Pilkey, W.D., Pilkey, D.F., Bi, Z. (2020). *Peterson's Stress Concentration Factors*, 4th ed. Wiley. doi:10.1002/9781119532552 (Ch. 2 Notches and grooves; Ch. 4 Holes).
- Liu, Q. et al. (2026). Geometry-informed neural operator transformer (GINOT). *CMAME* 451:118668 — benchmarks: elasticity, Poisson, bracket lug, micro-periodic unit cell, jet-engine bracket; boundary conditions supplied through extra encoders.
- Earlier references (Sukumar & Srivastava 2022; Berrone et al. 2023; Williams 1952; Noda & Takase 2004): see `docs/phase7_boundary_conditions.md`.
