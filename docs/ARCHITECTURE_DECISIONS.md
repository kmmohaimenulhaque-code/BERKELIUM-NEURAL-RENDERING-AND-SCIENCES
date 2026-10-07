# Architecture Decisions

Format: context → decision → consequences. Status: **Proposed** until the team accepts; **Accepted** after.

---

### ADR-001 — Authority separation between LLM and deterministic core — Proposed
**Context.** The brief requires engineering truth outside the LLM. Convention alone erodes (prompts, shortcuts).
**Decision.** Two schemas: `DesignProposal` (LLM-writable: intent, spec, structure, procedural geometry) and `DesignRecord` (adds evaluation, written only by core). Constrained decoding targets `DesignProposal`.
**Consequences.** The model cannot claim validation. Every number in the evaluation layer is reproducible from code + inputs.

### ADR-002 — Layered Design Record instead of a flat DesignGraph — Proposed
**Context.** Requirements, geometry ops and results differ in author, trust and lifetime.
**Decision.** Six layers (intent, specification, structure, geometry IR, evaluation, provenance); stable IDs; edits are JSON Patches.
**Consequences.** Same edit path for Studio sliders, LLM repairs and optimisers; edit history yields correction-pair data. Costs: patch semantics and migrations must be engineered carefully.

### ADR-003 — Serialisable expression language replaces code delegates — Proposed
**Context.** ShapeKernel modulations and CEM transforms are C# delegates — not inspectable, not LLM-authorable, not diffable.
**Decision.** Small typed expression AST with units (pint), parsed from strings, evaluated by NumPy (PyTorch later). Used for constraints, derived parameters, modulations and SDFs.
**Consequences.** Novel geometry becomes data. Expressivity is bounded by the language — extended deliberately, never via `eval`.

### ADR-004 — Multi-backend geometry: OCCT (exact) + Manifold (mesh) + own implicit; PicoGK optional — Proposed
**Context.** Need exact BREP/STEP for mechanical parts and robust fields/lattices for AM parts; target is Linux + MI300X.
**Evidence (this session, Linux, Python 3.13):** OCP 8.0.1 cube−sphere cut → `BRepCheck` valid, STEP written, 0.027 s. manifold3d 3.5.4 same boolean 0.008 s; gyroid level set 214 k triangles in 0.43 s. PicoGK runtime: only osx-arm64/win-x64 binaries; CMake has no Linux branch.
**Decision.** Backend-independent IR; OCCT is the exact authority; Manifold for mesh booleans/preview/level sets; Berkelium's own SDF evaluator for fields in phase 1. PicoGK revisited as an out-of-process worker only if a Linux runtime build succeeds and offers capability we lack.
**Rejected.** Blender as authority (mesh modeller, no engineering semantics). FreeCAD as kernel (GUI application wrapping OCCT). Single voxel kernel (no STEP, precision bound to voxel size).
**Consequences.** Representation is explicit per op; field → BREP never claimed. Measurements carry backend + tolerance (observed ~1 % volume gap OCCT vs faceted mesh).

### ADR-005 — Python core — Proposed
**Context.** LEAP 71 is C#. ML stack, OCP, Manifold, FastAPI (already used by Studio) are Python.
**Decision.** Core in Python ≥3.11, pydantic v2. Performance-critical kernels stay in native libraries; GPU field evaluation via PyTorch-ROCm when profiling justifies it.
**Consequences.** No direct PicoGK in-process use (C#); acceptable per ADR-004.

### ADR-006 — CEMs are pure, typed, versioned, and self-sampling — Proposed
**Context.** LEAP 71 CEMs use static mutable parameters, subclass-per-variant, and viewer side effects.
**Decision.** Protocol with `Requirements`, `Parameters`, `derive`, `check`, `expand`, `validators`, `analyses`, `sample`; no global state; entry-point registry; CEM without `derive` is labelled a *parametric template*.
**Consequences.** CEMs are testable, parallelisable, and double as dataset generators.

### ADR-007 — `not_evaluated` is a first-class validation status — Proposed
**Decision.** Validation results carry a fidelity kind (rule/geometric/analytic/numerical). No UI or API ever summarises a design as physically validated without numerical-fidelity results.

### ADR-008 — bf16 LoRA on MI300X before QLoRA — Proposed
**Context.** Brief says LoRA/QLoRA. Qwen3-32B bf16 ≈ 65 GB; MI300X = 192 GB. bitsandbytes ROCm support is uneven.
**Decision.** Default bf16 LoRA (PEFT + TRL on PyTorch-ROCm), serve with vLLM-ROCm. QLoRA only if memory measurements demand it.
**Consequences.** Simpler stack, no quantisation error during training. Must be confirmed by the smoke test (model load, 50 steps, memory profile).

### ADR-009 — Dataset labels are computed by the core; splits by family and held-out CEM — Proposed
**Decision.** Training targets come from CEM sampling, perturbation, procedural grammars, and verified teacher output. No unverified example enters a training split.
**Consequences.** Data generation depends on core → strong argument for keeping generators next to core (U1).

### ADR-010 — Studio is a client of a versioned Core API — Proposed
**Decision.** Studio consumes JSON Schemas, CEM parameter schemas, glTF artifacts and validation reports. LLM calls move from Studio backend into the Core model gateway. Electron must enable `contextIsolation` and disable `nodeIntegration` before rendering any model-generated content.

### ADR-011 — First milestone proves the spine with a gear pair, not RoverWheel — Proposed
**Context.** RoverWheel contains no engineering analysis and needs modulated-shape ops before it can be expressed. HelixHeatX needs the implicit backend and multi-domain validators.
**Decision.** M1 = schema + expressions + IR + OCCT/Manifold backends + L0–L4 (+L6) validators + **spur-gear-pair CEM** (closed-form engineering: involute geometry, minimum teeth for no undercut, contact ratio, Lewis bending stress; first assembly relation via centre distance) + **one hand-written procedural design** through the same IR + a first verified dataset slice. No LLM in the M1 loop; zero-shot Qwen baseline measured immediately after.
**Consequences.** Proves both creation mechanisms share one representation before investing in model training. Sequence after: M2 rocket nozzle (`derive` from isentropic relations), M3 HelixHeatX-class multi-domain implicit part, RoverWheel once modulated-shape ops exist.

---

## Open decisions

| ID | Question | Options | Recommendation |
|---|---|---|---|
| U1 | Repo layout | (a) core + training in one repo, Studio separate; (b) three repos | (a). Generators need core; schema needs one home. The current NEURAL-RENDERING repo could be repurposed or kept for its original research scope. |
| U2 | Production implicit backend | own SDF→GPU; PicoGK Linux worker; OpenVDB | Decide after M3 profiling |
| U3 | Is Qwen3-32B final? | Qwen3-32B; newer Qwen release; smaller model for `interpret` | Benchmark zero-shot first; check for newer releases at download time |
| U4 | Reasoning traces in training data | Train with thinking; without; mixed | Measure repair success both ways |
| U5 | Inference location | Fireworks (hosted LoRA) vs vLLM on MI300X | Keep both behind gateway; depends on MI300X duration (R11) |
| U6 | Dataset storage/versioning | HF Hub private, DVC, plain manifests + object store | Manifests first; tool later |
| U7 | Studio form factor | Electron vs web | Web-first API; Electron as wrapper |
| U8 | Licences for team repos | Apache-2.0, MIT, proprietary | Decide before external contributions |
| U9 | Ownership | Who owns core / Studio / training | Team decision |

### ADR-012 — Own unit system inside the core; pint only at boundaries — Accepted (implementation)
**Context.** ADR-003 named pint. Inside hashed, dataset-generating code we need deterministic, hashable,
dependency-light units.
**Decision.** `berkelium.units`: SI magnitude + integer dimension vector; angles dimensionless (SI); unit
strings with `* / ^`. pint may be used for I/O but is not a core dependency.
**Consequences.** Removes a dependency; unit table grows deliberately (offset temperatures unsupported).

### ADR-013 — This repository is the core — Accepted
**Decision (resolves U1).** `BERKELIUM-NEURAL-RENDERING-AND-SCIENCES` hosts core + datasets + training.
The 105 zero-byte tool-named placeholder files were removed (no content); tools are now real dependencies
in `pyproject.toml`. Docs moved to `docs/` with underscore names.

### ADR-014 — Gear CEM scope decisions — Accepted (implementation)
- 20° full-depth only in v0.1 (sourced rack + Lewis table). Undercut gears are rejected, not generated.
- Profile-shift total from the KHK inverse calculation; split equally unless the pinion needs more (designer choice, recorded as an assumption).
- Lewis only for x = 0, 12–400 teeth; otherwise `not_evaluated`. Contact ratio: fail < 1.0 (KHK), warn < 1.5 (Shigley §14-1). No unsourced thresholds.
- Backlash = designer-chosen tooth thinning, not a standard.

### ADR-015 — One spline definition for all backends — Accepted
IR splines are chord-length natural cubics (dense point lists are curve samples). OCCT interpolates the same samples; Manifold uses them as a polyline. Lantern volumes agree within 0.04 %.

### ADR-016 — Physics stack: scikit-fem in-process; Gmsh, CalculiX, OpenFOAM out of process — Accepted (implementation)
**Context.** The core license is still undecided (ADR in SOURCES ledger). FEniCSx 0.11 (DOLFINx LGPL-3.0) is the
strongest Python FEM but installs via conda/apt/Docker and pulls PETSc/MPI; Gmsh's Python module is a ctypes
binding of a GPL-2.0-or-later library; CalculiX (GPL-2.0) and OpenFOAM (GPL-3.0) are executables.
**Decision.** In-process FEM = scikit-fem (BSD-3, pure Python + SciPy). Gmsh, CalculiX, OpenFOAM are invoked only
as separate executables over files; they are never imported. FEniCSx stays a candidate adapter for large/parallel
problems (out of process or optional). **Consequences.** Licence-clean core; small direct solves (3 GB / 1 CPU
handles ~10^5 dofs); scale-up path is an adapter, not a rewrite.

### ADR-017 — Estimates, not numbers: error-carrying quantities and three-valued requirements — Accepted
**Decision.** Every computed physical quantity is an `Estimate` (value, fidelity, status, discretisation error,
model-form error, method). Numerical error bands come from a per-QoI Celik/Roache GCI study. L7 requirement checks
evaluate the whole interval: pass / fail / **indeterminate**. A design is `physically_validated` only with passing
L7 numerical evidence and no indeterminate/failed L7 results — enforced by the schema, not by convention.
**Why.** It is the one mechanism shared by every domain (FEA, heat, CFD, reduced-order, analytic): it makes
"how sure are we?" a typed, checkable, trainable object, and it gives the repair loop a third outcome
("refine the mesh / add evidence") that pass/fail hides.

### ADR-018 — Regions are level sets, evaluated on vertices — Accepted
**Decision.** Boundary conditions attach to `Region`s defined by an implicit-field expression (the existing
expression language, mm) over facet vertices and normals — never to mesher facet ids. **Why.** Works for any
mesher/backend, survives remeshing and refinement (required for GCI), is model-proposable and diffable, and
reuses the implicit-geometry convention (f <= 0 is inside). Vertices, not centroids, because centroids of curved
faces sag inside the surface by ~h^2/8R (this broke the first radial-conduction run).

### ADR-019 — The Evidence Calculus: claims resolved over a hierarchy of executable laws — Accepted (experimental)
**Hypothesis.** Analytic formulas, correlations, FEM/CFD, surrogates and measurements are one kind of object:
an evidence source mapping a design point to an interval-valued estimate, valid only inside a declared domain,
at a declared cost. Engineering questions are claims; deciding one is search over evidence. Multi-fidelity,
capability reasoning, falsification and experiment-awareness then need no per-domain machinery.
**Design.** `berkelium.evidence`: `Law` (typed executable model + dimension-checked validity predicates in the
safe expression language + named dimensionless groups + cost + fidelity), `Claim`, `Est` (interval; unknown
model-form error = corroboration only), `resolve` (cheapest decisive source; escalate on straddle; skip and
record out-of-domain; any two disjoint known intervals = CONFLICT, never a decision), `calibrate`/`calibrated`
(a cheap law's model-form bound learned from verified high-fidelity runs, valid only inside the sampled box).
**Experiment E1** (`scripts/evidence_experiment.py` -> `docs/experiments/evidence_e1.json`, deterministic):
see IMPLEMENTATION_STATUS. **Evidence.** Same minimum design from FEM-only and evidence-resolved search;
conflicts detected for a contradicting measurement and a deliberately mis-implemented law; out-of-domain
points escalate; transitional pipe flow returns insufficient_evidence; the pipe domain needed zero new
resolution code. **Limitations.** (1) The learned bound is an empirical max over n=8 samples — not a
statistical guarantee; final designs must still be verified at the reference fidelity (E1 does this).
(2) Calibration cost is amortised only across reuse: one search alone costs more (8 + 2 vs 9 FEM runs).
(3) Intervals are worst-case sums; no probabilistic propagation of parameter uncertainty yet. (4) Laws are
Python callables; their *formulas* are not yet symbolic objects that can be composed or differentiated.
**Next abstraction.** (a) Laws as symbolic, composable equation objects in the expression language (so CEMs
become compositions of laws + constraints, and sensitivities come for free); (b) sample-size-aware bounds
(e.g. conformal-style inflation) and parameter-uncertainty propagation; (c) every Resolution persisted as
engineering memory and reused as calibration data; (d) the LLM proposes claims, laws and decompositions —
the calculus decides.

### ADR-020 — Acausal law networks: relations are the primitive, computations are derived — Accepted (experimental)
**Descent.** ADR-019 asked "which evidence decides a claim?" but its Laws were opaque Python callables. Underneath
every CEM and every analytic Law is a set of *undirected relations* between dimensioned quantities. Which variable
is computed from which is not engineering knowledge — it follows from what is known in a given problem.
**Hypothesis.** If engineering knowledge is stored as dimension-checked acausal relations (+ validity, fidelity,
model-form bound, references, assumptions), then sizing, checking, inversion, sensitivity, uncertainty propagation,
constrained optimisation, specification diagnostics and cross-domain composition all come from ONE domain-free
engine — and a new domain is added by writing relations, not code.
**Design.** `berkelium.laws`: Var, Relation ('a == b' | 'a <= b' | 'a >= b' in the safe expression language),
LawSet (+ compose by shared names). `solve` = bipartite matching of unknowns to equations (the structural analysis
used by acausal modelling languages) -> Tarjan SCC block-lower-triangular order -> per block: scan + Brent with ALL
roots found (ambiguity is reported, never silently picked) or a coupled Newton-type solve -> redundant equations
become consistency checks (CONFLICT) -> inequalities and validity predicates checked. Partial solutions when
underdetermined, naming the free degrees of freedom. `sensitivities` (log-log elasticities), `propagate`
(first-order + seeded Monte Carlo), `optimise` (grid-seeded bounded Nelder-Mead with constraint penalty, reports
active constraints), `as_law` (a LawSet becomes an Evidence-Calculus Law; unknown model-form bounds propagate as
unknown).
**Experiment E2** (`scripts/laws_experiment.py` -> `docs/experiments/laws_e2.json`). Four domains as pure data
(cantilever beam, Lame thick cylinder, isentropic nozzle, insulated wall) + one cross-domain composition (rocket
chamber = nozzle + vessel + 2 interface relations). Results in IMPLEMENTATION_STATUS.
**Limitations.** Scalar algebraic relations only (no fields, ODE/PDE, tensors, vector quantities); scalar blocks use
a 400-point scan (roots closer than the scan spacing can be missed); optimiser is local after a coarse grid (no
global guarantee); Monte Carlo assumes independent Gaussian inputs; `as_law` treats each relation's model-form bound
additively; elasticities are undefined at zero values. Relations are not yet symbolically differentiated (finite
differences of re-solves).
**Next abstraction.** (1) Field relations: let a Relation's term be a *discretised field functional* (an FEM QoI),
so analytic and numerical models share one network — the solver then chooses fidelity per relation. (2) ComposedCEM:
a LawSet + a Geometry-IR template whose parameters are network variables, registered as a CEM, replacing
hand-written CEMs. (3) Relations as training data: (problem, known set, derived plan, diagnostics) are verified
reasoning traces for the model.

### ADR-021 — Engineering memory + gated relation discovery — Accepted (experimental)
**Hypothesis.** Competence accumulates if verified computations are stored as computable records and if new
relations can be *discovered* from them — provided promotion into knowledge is gated by independent falsification.
**Design.** `berkelium.memory`: append-only, content-addressed records (evidence | relation | rejection |
resolution); memoisation = memory. `berkelium.discovery`: Buckingham-Pi groups derived from units (exact rational
nullspace); exhaustive sparse monomial regression on a TRAIN family; Occam selection on a HELD-OUT family;
physics limit checks (declared limits falsify candidates before fitting); a Gate (held-out residual <= k x
evidence error + tol, minimum sample sizes); promoted relations carry a validity box and a model-form bound.
**Experiment E3** (`scripts/discovery_experiment.py`; evidence in `docs/experiments/memory_e3.jsonl`, results in
`discovery_e3.json`): 40 verified FEM beams (train nu in {0.2, 0.4}, held out nu = 0.3, plus 10 fresh benchmark
points). **Evidence.** (1) The data-only gate PROMOTED a law with a term that does not vanish in the slender limit —
a physically wrong law that fits the data. Declaring the limit "Euler-Bernoulli is exact as h/L -> 0" falsified
3534 of 4095 candidates; the promoted law is 0.6300270715359171*(h/L)^2 + -0.07530198566629838*(h/L)*(h/b)^-1 (held-out max |residual| 4.65e-03 vs
evidence-based limit), validity box h/L in [0.05, 0.2], h/b in [1, 2], nu in [0.2, 0.4], model-form bound
1.81%. (2) A deliberately impoverished hypothesis space was rejected by the same
gate. (3) On 10 fresh claims set 0.3-4 % from the FEM truth: FEM runs 10 -> 6,
verdict disagreements 0, conflicts 0. (4) At h/L = 1/3 (outside
the box) the discovered law refuses and the claim is INSUFFICIENT_EVIDENCE without FEM.
**Limitations.** One quantity, one geometry family; monomial library only; the gate's k and tol are policy, not
derived; held-out family shares geometry type with training; the promoted law is a surrogate valid in its box —
its b/L term is fitted, not explained. Data-only gates are demonstrably insufficient; which physical checks to
declare is itself knowledge the system does not yet discover.
**Next abstraction.** Put the LLM inside the loop as hypothesis generator (relations, limits, decompositions)
with the core as falsifier; record agent trajectories (accepted AND falsified) as training data; discover
dimensionless structure for multi-quantity / multi-domain relations.

### ADR-022 — ClaimEnv: an agentic engineering environment whose reward requires grounding — Accepted (experimental)
**Hypothesis.** To learn HOW to engineer (not how to describe engineering), a policy must act through evidence
tools and be rewarded for grounded decisions, not for matching answers. An answer-matching metric cannot tell a
grounded engineer from a fluent guesser.
**Design.** `berkelium.agent`: `ClaimEnv` (observation = claim, design point in SI with dimension vectors, tool
catalogue with fidelity/cost/validity, gathered evidence; actions = evaluate(tool) | declare(verdict); reward
version `claimenv-r1`: grounded correct +1 - cost term; ungrounded pass/fail -1 even if correct; wrong grounded -2;
lazy abstention -0.5; justified abstention +0.5). Baselines: AlwaysHighest, CheapestSufficient (the evidence
calculus as a policy), Overconfident (declares from a point estimate). `GatewayPolicy` lets Qwen act via the
model gateway (JSON actions) — implemented, NOT yet run (no endpoint here).
**Experiment E4** (`scripts/agent_experiment.py` -> `agent_e4.json`, `trajectories_e4.jsonl`): 240 tasks = 40
verified FEM points x 6 claim margins (0.3-3 %); truth = verified FEM verdict (115 pass, 115 fail, 10 undecidable).
**Evidence.** Overconfident: return -1.0 on all 240 although its verdict was correct in 206 (86 %) — an
accuracy-only benchmark would have scored it 86 %. Grounded policies: 230 correct, 10 honest abstentions, 0 wrong.
Knowledge promoted into memory by E3 (the discovered relation, rebuilt from its memory record) cut
CheapestSufficient's mean cost 1001 -> 715 overall, and 1001 -> 670 (-33 %) on the 60 tasks at the 10 points NOT
used for discovery, with identical (all-correct) outcomes there.
**Limitations.** One quantity; scripted policies; reward weights are policy choices; tasks share one geometry
family; no LLM policy measured yet; truth inherits FEM discretisation error (handled by the 10 undecidable tasks).
**Next.** Run GatewayPolicy (base Qwen3-32B vs Berkelium LoRA) on ClaimEnv; use trajectories_e4.jsonl for SFT and
(preferred, rejected) pairs from grounded vs ungrounded episodes for preference optimisation; add law-network
construction actions so the agent must also BUILD the model it then verifies.

**ADR-022 addendum (found while preparing MI300X runs).** (1) `ClaimEnv.observe()` returned the live evidence list,
so every stored trajectory observation showed evidence gathered LATER in the episode (a label leak that would have
poisoned SFT prompts). Fixed (observations copy the list); E4 scores are unchanged because policies act on the live
observation, and trajectories were regenerated. (2) The first family split was leaky (fresh points with nu near 0.3).
Families are now: train = nu in {0.2, 0.4}; heldout = every other nu. Training data (SFT 386 train / 61 val, 120
preference pairs) comes only from the train family, built by `scripts/agent_build_training.py`; the LLM evaluation
runner is `scripts/agent_llm_eval.py`; procedure in `docs/MI300X_AGENT_RUNBOOK.md`. No LLM result exists yet.
