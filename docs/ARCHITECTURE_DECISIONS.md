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
