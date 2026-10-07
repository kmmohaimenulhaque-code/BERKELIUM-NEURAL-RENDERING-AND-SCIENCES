# Implementation status

Branches: `main` (docs only) <- `core/m1-foundation` (M1, head 8a26f52) <- `physics/p1-numerical` (this milestone).
Vocabulary: **implemented** (code exists) · **tested** (unit tests) · **analytically verified** (matches a closed-form
oracle exactly) · **numerically verified** (true error measured against an exact solution and found inside the
computed discretisation-error band) · **not_evaluated** · **unsupported** · **experimental**.

## Model / data artefacts (state as of this branch)
| Artefact | State |
|---|---|
| Qwen3-32B LoRA adapter | trained by the owner on MI300X; published at huggingface.co/Mhaquehaque/berkelium-qwen3-32b-lora (PEFT 0.21.2, base Qwen/Qwen3-32B). Model card is still the empty template: **no eval metrics or run manifest are published** |
| Verified M1 dataset | published at huggingface.co/datasets/Mhaquehaque/berkelium-engineering-dataset (gear-only, manifest hash 29d0fb09... for seed 0 in the owner's run) |
| Zero-shot vs LoRA comparison | **not recorded in the repo** — `out/eval/base.json` / `lora.json` must be committed or summarised before PHYSICS-9 |
| Prompt drift | the plan prompt lists every registered CEM; registering `structure.cantilever_beam@0.1` changes it. Compare base vs LoRA on the SAME prompt version (re-run both), or the comparison is confounded |

## Physics (PHYSICS-1..7)
| Area | Module | Status |
|---|---|---|
| Physics schemas: Material, Region (level-set selector), 10 load/BC kinds, MeshSpec, SolverSpec, ConvergenceSpec, QoI, AnalysisCase; results: Estimate, ConvergenceStudy, MeshStats, SimProvenance, SimulationResult | `physics.schema` | implemented, tested (dimension checks, cross-reference checks) |
| Semantic regions: facets selected by an implicit-field expression evaluated at facet VERTICES (mesher-independent) | `physics.mesh` | implemented, tested |
| Meshing: structured tet box (exact nested refinement); Gmsh 4.15.2 **out of process** from OCCT STEP; quality metrics, mesh hashes, systematic size+curvature refinement | `physics.mesh` | implemented, tested |
| Linear static elasticity (P1/P2 tets): fixed, prescribed displacement, traction, surface force (uniform, no point loads), pressure, body acceleration; displacement, von Mises (domain or surface), reactions; residual + discrete equilibrium per solve | `physics.solvers.fem` | **numerically verified** (Lame) + analytically verified (axial bar) |
| Steady conduction (P1/P2): temperature, heat flux, convection, volumetric source; boundary heat flows, discrete heat balance | `physics.solvers.fem` | **numerically verified** (radial conduction) + analytically verified (slabs) |
| Solution verification: Celik (2008) GCI with non-constant ratio, conservative order cap, per-QoI convergence status | `physics.verification` | tested on synthetic sequences; used by every numerical case |
| Reduced-order pipe flow (Darcy-Weisbach; laminar exact; Colebrook with +-15 % model-form band; transitional = non_converged) | `physics.fluids` | analytically verified (laminar), tested (Colebrook) — **fidelity reduced_order, not CFD** |
| CFD adapter protocol; OpenFOAM adapter | `physics.fluids` | interface only: always `unsupported` (detection only) |
| L7: solution-verification results + three-valued requirement checks on error intervals (`sim.<case>.<qoi>`); `indeterminate` status; `physically_validated` schema-gated on L7 numerical evidence | `physics.l7`, `validation`, `schema.evaluation` | implemented, tested end-to-end |
| Pipeline analyses stage: component -> OCCT -> STEP -> Gmsh -> FEM -> GCI -> L7 -> DesignRecord (+ VTU field artefact); records byte-reproducible | `pipeline` | implemented, tested |
| Material library (4 entries, each cited; nominal class values) | `physics.materials` | implemented |
| `structure.cantilever_beam@0.1` CEM (EB sizing from deflection + stress, slenderness check, default 3-D analysis case) | `cem.library.beam` | implemented, tested end-to-end |
| CalculiX cross-check adapter, thermal-structural coupling, nozzle / pressure-vessel / heat-exchanger CEMs, physics dataset slice | — | **not started** |

### Numerical evidence (finest level; reproduced by `tests/test_physics.py`, `tests/test_physics_pipeline.py`)
| Case | Exact | Computed | True error | GCI band | Observed order |
|---|---|---|---|---|---|
| Radial conduction, tube 10/20/10 mm, P2, Gmsh 3 levels | Q = 453.236 W | 452.948 W | -0.064 % | +-0.352 W (covers) | 2.00 |
| Lame plane strain, quarter tube, p = 10 MPa, P2, 3 levels r=1.5 | u_r(a) = 9.5333e-4 mm | 9.5224e-4 mm | -0.115 % | +-1.32e-6 mm (covers) | 2.13 |
| same, surface-mean von Mises at r = a | 23.1325 MPa | 23.0775 MPa | -0.238 % | +-0.105 MPa (covers) | 1.35 |
| Cantilever 100x10x10, structured P2, 3 levels | Timoshenko 0.20153 mm (model, not exact) | 0.19993 mm | -0.79 % vs model | +-2.3e-4 mm | 2.13 |
| Axial bar / heated slab / convective slab | closed form | exact to 1e-12 | - | exact reproduction | - |
| Laminar pipe | Hagen-Poiseuille | exact to 1e-12 | - | - | - |
Domain-max von Mises at quadrature points converged in the Lame case (p = 1.80) but is a weak QoI: at re-entrant
corners (clamped beam root) it is singular and the GCI study correctly reports `non_converged`.

## Agentic environment (ADR-022) — experimental
| Area | Module | Status |
|---|---|---|
| ClaimEnv, reward claimenv-r1, baseline policies, GatewayPolicy (LLM) | `berkelium.agent` | implemented, tested; LLM policy **not evaluated** (needs endpoint) |
| Experiment E4 + 1440 labelled trajectories | `scripts/agent_experiment.py` | run; results committed |

## Engineering memory + discovery (ADR-021) — experimental
| Area | Module | Status |
|---|---|---|
| Content-addressed append-only memory (evidence, relation, rejection, resolution); memoisation | `berkelium.memory` | implemented, tested |
| Buckingham-Pi groups from units; sparse monomial discovery; limit falsification; promotion gate | `berkelium.discovery` | implemented, tested (planted-law recovery, gate rejections) |
| Experiment E3 | `scripts/discovery_experiment.py` | run; results + 40 evidence records committed |
See `docs/AGI_ENGINEERING.md` for the capability ladder and measured position.

## Acausal law networks (ADR-020) — experimental
| Area | Module | Status |
|---|---|---|
| Var / Relation / LawSet / compose; structural matching + BLT; all-roots scalar solve; coupled solve; conflict, ambiguity, underdetermination, validity diagnostics; sensitivities; UQ; optimise; as_law | `berkelium.laws.core` | implemented, tested |
| Domains as data: cantilever beam, Lame thick cylinder, isentropic nozzle, insulated wall; rocket chamber composition | `berkelium.laws.library` | implemented, tested against independent references |

E2 results (references are independent of the machinery):
nozzle A/A* (M=2, gamma=1.4) 1.687500 vs NACA 1135 1.6875; p/p0 0.127805 vs 0.1278 · eps = 1.6875 inverted -> AMBIGUOUS
{0.3722, 2.0000}; declaring M_e >= 1 -> 2.0000 · beam minimum-mass height 11.696071 mm vs closed form 11.696071 and the
hand-written CEM's unrounded 11.696071, active constraint = strength · elasticities of deflection
{P 1, L 3, E -1, b -1, h -3} to 1e-9 · first-order relative sigma 0.07810 = analytic 0.07810, Monte Carlo (n=400) 0.0784 ·
insulation thickness 31.000 mm vs closed form 31.000 · rocket chamber (1 kN, 2 MPa, gamma 1.2) solved by an automatically
derived 17-block plan from 18 relations, of which 2 are new · Lame network sigma_vm 23.1325 MPa lies inside verified FEM
[22.972, 23.183] MPa (no conflict); a thin-wall formula misused at ro/ri = 2 conflicts with FEM and Lame; a 50 % Poisson-
ratio error (23.096 MPa) is NOT detectable by FEM at this precision (inside its band) — only by the exact network.
That last result is an identifiability limit, reported as such.

## Evidence calculus (ADR-019) — experimental
| Area | Module | Status |
|---|---|---|
| Law / Claim / Est / Resolution, resolve (escalation, out-of-domain skip, conflict detection), calibrate | `berkelium.evidence.core` | implemented, tested (synthetic laws + real FEM) |
| Beam laws (Euler-Bernoulli, Timoshenko, FEM-P2-GCI); pipe laws (Hagen-Poiseuille, Colebrook) | `berkelium.evidence.library` | implemented, tested |
| Experiment E1 | `scripts/evidence_experiment.py` | run; results committed (`docs/experiments/evidence_e1.json`) |

E1 numbers (E = 200 GPa, nu = 0.3, P = 500 N, L = 200 mm, h/b = 2, limit 1.0 mm):
calibration of Euler-Bernoulli vs FEM on 8 beams (L/h 6..20, h/b 1..2) -> learned relative bound 1.394 %;
minimum height 20.0781 mm from BOTH FEM-only bisection (9 FEM runs) and evidence-resolved bisection (2 FEM runs:
one straddle escalation + final verification). Including calibration, one search costs 10 vs 9 FEM runs; k reused
searches cost 8 + 2k vs 9k. Falsification probes: contradicting measurement -> conflict; law with a 25 % bug ->
conflict; L/h = 2.9 -> EB skipped (outside validity), FEM decides. Pipe: laminar -> Hagen-Poiseuille, turbulent ->
Colebrook, transitional -> insufficient_evidence.

## M1 foundation
| Area | Module | Status |
|---|---|---|
| Units, expression language (AST, Pratt parser, dimension-checked eval, NumPy/closure field compile, no eval) | `berkelium.units`, `berkelium.expr` | done, tested |
| Canonical schemas (DesignProposal, DesignRecord, all layers), JSON Schema export, RFC 8785 hashing, migrations | `berkelium.schema` | done, tested |
| Geometry IR (primitives, profiles, extrude/revolve, booleans, transforms, patterns, implicit fields, tags, deps, expression args) | `berkelium.geometry.ir` | done, tested |
| OCCT exact-BREP backend (deterministic STEP; lossy-flagged STL/GLB) | `geometry.occt` | done, tested (OCP 8.0.1, 7.x imports supported) |
| Manifold mesh backend + level sets; capability planner (field → STEP refused) | `geometry.manifold_backend`, `geometry.backend` | done, tested |
| Validation L0–L4, L5 (envelope; other checks `not_evaluated`), L6 gear, L7 always `not_evaluated` | `berkelium.validation`, `manufacturing` | done, tested |
| CEM protocol, registry (entry points `berkelium.cems`) | `berkelium.cem.protocol` | done |
| Spur gear pair CEM (sourced formulas, rack-simulated involute + trochoid fillet, derive, checks, mesh-interference) | `cem.library.gear` | done, oracle-tested |
| Creative procedural designs (lantern BREP, gyroid orb field) | `berkelium.designs` | done, tested |
| Deterministic pipeline → content-hashed DesignRecord, artifact store, job stages | `berkelium.pipeline` | done, tested |
| Dataset factory (plan / repair / procedural; verified; family splits; manifests; rejected log) | `berkelium.datasets` | done, tested |
| Model gateway (OpenAI-compatible vLLM/Fireworks, replay), orchestrator + JSON-Patch repair loop, evaluation harness | `berkelium.ai` | done, tested offline (fake server) |
| SFT rendering + length audit | `berkelium.training.data` | done, tested |
| MI300X smoke test, LoRA SFT (PEFT + transformers.Trainer, shared tokenisation/masking in `training/common.py`), vLLM install/serve scripts | `training/`, `scripts/` | CPU dry-run passed end-to-end with Qwen3-0.6B (smoke + 2-step SFT + adapter save); **not yet run on MI300X** |
| Endpoint robustness: transport errors → `GatewayError`, loopback bypasses proxies, `berkelium model-check --wait`, eval preflight + abort after 3 consecutive endpoint failures (no fake 0 % baseline) | `berkelium.ai`, `berkelium.cli` | done, tested |
| Core API v1 (schemas, CEMs, designs, JSON-Patch revisions, validation, artifacts, intents) + CLI | `berkelium.api`, `berkelium.cli` | done, tested |
| Zero-shot Qwen3-32B baseline | `berkelium eval` | **blocked**: needs a Qwen3-32B endpoint (MI300X vLLM or Fireworks) |
| Studio integration | — | not started |

## Run order on the MI300X

Two environments: `.venv` (Berkelium + training, keeps your ROCm torch) and `.venv-vllm` (vLLM's own pinned torch).
vLLM and training cannot share the GPU: vLLM reserves 90 % of VRAM until it exits.

```bash
# 0. setup (training venv)
source .venv/bin/activate
pip install -e ".[occt,mesh,api,dev,train]"
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"  # must still be +rocm
pytest
berkelium dataset --out out/ds --seed 0 --n-valid 400 --n-boundary 100 --n-invalid 200 --n-procedural 30
berkelium sft-render --dataset out/ds --out out/sft
scripts/install_vllm_rocm.sh            # once; or VLLM_MODE=docker VLLM_IMAGE=vllm/vllm-openai-rocm:<tag>

# 1. zero-shot baseline
mkdir -p out/logs
scripts/serve_vllm.sh > out/logs/vllm-base.log 2>&1 &
export BERKELIUM_MODEL_URL=http://127.0.0.1:8000/v1 BERKELIUM_MODEL=Qwen/Qwen3-32B
berkelium model-check --wait 1800      # first run downloads ~65 GB, then loads
berkelium eval --dataset out/ds --label base --out out/eval/base.json
pkill -f "vllm serve"; sleep 20

# 2. smoke test, then 3. training (refuse to start if < 120 GB VRAM free)
python training/smoke_test.py --config training/configs/qwen3_32b_lora.yaml --data out/sft/train.chat.jsonl --steps 10 --out training/runs/smoke.json
python training/sft_lora.py --config training/configs/qwen3_32b_lora.yaml

# 4. post-training eval
scripts/serve_vllm.sh training/runs/qwen3-32b-lora > out/logs/vllm-lora.log 2>&1 &
export BERKELIUM_MODEL=berkelium
berkelium model-check --wait 1800
berkelium eval --dataset out/ds --label lora --out out/eval/lora.json
pkill -f "vllm serve"
```

Exit code 3 from `model-check`/`eval` means the endpoint is not ready or was lost; no report is written.
A valid eval report has `transport_error_rate: 0`.
