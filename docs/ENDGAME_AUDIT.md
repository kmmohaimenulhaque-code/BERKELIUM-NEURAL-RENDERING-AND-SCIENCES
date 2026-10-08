# Endgame audit — 2026-10-08 (branch physics/p1-numerical @ c56b37e)

Labels: **IMPLEMENTED** (code exists, tests pass) · **TRAINED** (an adapter was optimised on it) · **EVALUATED**
(an LLM was scored on a held-out task benchmark and the report is committed) · **NOT IMPLEMENTED**.

## 1. Repository + artifact integrity (checked, not assumed)
| Check | Result |
|---|---|
| Branch contains all Claude commits up to db023f7 | yes; then 4fd24fa (setup_mi300x.sh), c56b37e (datasets) |
| `out/agent`, `out/agent_v2` committed datasets | present; hashes match their manifests; **byte-identical to a fresh rebuild** |
| Held-out leakage in AGENT-V2 data | 0 rows from vessel / gas_dynamics / rocket; 0 rows with nu = 0.3 |
| Training configs vs HF run manifests | all four `config_sha256` match `training/configs/*.yaml` exactly |
| Training data vs HF run manifests | M1 = HF dataset sft/train (c947c690…); V1 = out/agent (be8c845c…); V2 = out/agent_v2 (a80ba9c2…); DPO pairs = out/agent_v2/pairs (82f173ec…) |
| Test suite at c56b37e | non-physics suite green (physics code unchanged since its last green run) |
| **NOT committed** | `out/eval/*.json` (no LLM evaluation of ANY generation), `out/env_manifest_mi300x.json`, `training/runs/smoke*.json`, run manifests (they exist only on HF), `core/m1-foundation` still contains a stray uploaded `berkelium-physics-p1.bundle` |
| Environment drift | M1 trained on torch 2.10.0+rocm7.14 (HIP 7.14); V1/V2/DPO on torch 2.10.0+rocm7.0 (HIP 7.0) — two different stacks, no committed env manifest for either |

Model lineage, hyper-parameters and loss curves: `docs/manifests/model_generations.json`.

## 2. What each generation was trained to do
| Generation | Data | Learned behaviour (output format) | Optimisation evidence |
|---|---|---|---|
| BASE | — | Qwen/Qwen3-32B | — |
| M1 (r16) | 574 gear plan/repair rows | NL request -> `DesignProposal` JSON (gear CEM / procedural) and JSON-Patch repairs | eval_loss 0.0692 (2 epochs) |
| AGENT-V1 (r16) | 386 ClaimEnv steps (nu 0.2/0.4) | ClaimEnv obs -> one action: evaluate(tool) / declare(verdict), cheapest-sufficient grounded evidence | eval_loss 0.0100 (3 epochs) |
| AGENT-V2 (r32) | 906 = V1 rows + 520 ModelEnv steps (structures, heat, fluids) | + ModelEnv obs -> search / add / remove / solve / declare: build a law network from a context-tagged relation library | eval_loss 0.0022 (3 epochs) |
| AGENT-V2-DPO (r32) | 204 env-labelled pairs, init = V2, beta 0.1, 25 steps | prefer grounded action over ungrounded / first-match-retrieval action at the same observation | loss 0.695 -> 0.137, train pref-acc 1.0, margin 19.2 |

Caveats: eval_loss is teacher-forced token loss on IN-DISTRIBUTION validation rows of highly templated JSON; it does
not measure task success. DPO preference accuracy is on its own training pairs (no held-out pairs); margin 19 after 25
steps suggests the pairs are easy/separable, not that behaviour generalises. **M1 prompt drift**: M1 was trained when the
plan prompt listed only `gear.spur_pair`; the current prompt also lists `structure.cantilever_beam` — evaluate M1 with
the prompt it was trained on, or the result is confounded.

## 3. Learned vs deterministic
| LLM (learned, proposes) | Deterministic core (authority) |
|---|---|
| intent -> DesignProposal (M1) | schemas, units, expression language, dimension checks, quarantine |
| which evidence tool to call, when to declare (V1) | evidence calculus: intervals, validity, conflict, insufficient_evidence |
| which relations to add/remove, when to solve/declare (V2, DPO) | law engine: matching, BLT solve, all roots, closure, competition |
| repair patches (M1) | CEM derivations, Geometry IR, OCCT/Manifold/implicit realisation, measurement |
| — | Gmsh meshing, FEM (elasticity, conduction), GCI, L7, records, hashes |
| — | memory, discovery gate, reward/judging, dataset builders |
No learned component writes a measurement, a simulation result or a verdict that is accepted without the core.

## 4. What has actually been demonstrated
EVALUATED (deterministic policies only): E1 evidence calculus, E2 law networks, E3 discovery, E4 ClaimEnv (CheapestSufficient
230/240 correct + 10 honest abstentions, 0 wrong), E5 ModelEnv (constructor 68/68, greedy retrieval 28/68).
TRAINED but NOT EVALUATED: M1, AGENT-V1, AGENT-V2, AGENT-V2-DPO. **There is currently no evidence that any adapter beats
BASE on any task.** The eval runners exist (`scripts/agent_llm_eval.py`, `scripts/model_llm_eval.py`, `berkelium eval`).

## 5. Gaps for natural language -> spec -> verified model -> 3D geometry/mesh
| Link | Status |
|---|---|
| NL -> DesignProposal -> CEM -> Geometry IR -> STEP/STL -> L0-L7 (gear, beam, procedural) | IMPLEMENTED; M1 trained (gear only); not evaluated |
| NL -> structured *Problem* (context facts, knowns, target, requirements) for ModelEnv | **NOT IMPLEMENTED** (ModelEnv starts from a structured problem) |
| law network -> verdicts with evidence | IMPLEMENTED (V1/V2 trained, not evaluated) |
| solved law network -> geometry (variables bound to a parametric template) | **NOT IMPLEMENTED** except the hand-written beam CEM |
| geometry -> analysis case (regions/BCs) generated from the same template -> L7 cross-check of the law value | beam only (`default_analysis`) |
| one runtime chaining these with stops for underdetermined / contradictory / insufficient evidence | **NOT IMPLEMENTED** (three separate entry points: `berkelium design`, ClaimEnv, ModelEnv) |
| Studio UI (chat, 3D viewer, validation panel) | NOT IMPLEMENTED (Studio frontend is the Vite template; Electron security fix pending) |

## 6. Target runtime loop (what one request will execute)
1. user text -> **LLM intent compiler** -> Problem/DesignProposal JSON (schema-validated, units checked)
2. core: missing/contradictory spec? -> ask the user (no guessing)
3. **LLM agent (V2-type actions)** builds the law network; core `solve` verifies (determined / underdetermined / contradictory / outside validity / ambiguous)
4. core evidence calculus decides each requirement claim: analytic first, FEM when intervals straddle, CONFLICT surfaced
5. core **geometry binding**: solved variables -> template -> Geometry IR -> OCCT STEP (exact) / Manifold STL / implicit mesh
6. core: same template -> regions + BCs -> Gmsh -> FEM -> GCI -> L7 on the REAL geometry (cross-checks the law network)
7. DesignRecord + artifacts (STEP, STL/GLB, VTU) + provenance; failures -> LLM repair patch -> back to 3
8. verified trajectory -> memory + training data (next generation)

## 7. Shortest sound path (in order; each step is independently testable)
1. **Evaluate what exists (GPU, no new code)**: BASE, M1 (gear-only prompt), V1, V2, V2-DPO on ClaimEnv held-out, ModelEnv held-out + seen, and `berkelium eval` for M1. Commit every `out/eval/*.json`.
2. **Geometry binding** (deterministic): per context family a template (rect cantilever -> box; cylinder vessel -> tube; nozzle -> revolved profile from d_t, d_e, CR, ri, ro) emitting Geometry IR + an AnalysisCase. Generalises the beam CEM. Test: law value inside the FEM GCI band.
3. **Intent compiler data**: deterministic NL renderings of ModelEnv/ClaimEnv problems -> Problem JSON (labels by construction, round-trip verified, held-out domains preserved).
4. **Runtime** `berkelium.agent.runtime` + API `/v1/sessions`: steps 1-8 above, constructor/CheapestSufficient as deterministic fallback, every stop explicit.
5. **AGENT-V3**: one adapter on end-to-end trajectories (compile + construct + evidence + geometry + repair) + curriculum from step-1 failures; DPO only with held-out preference eval.
6. Studio: chat + 3D viewer (STEP->GLB) + validation/evidence panel over the API; Electron contextIsolation on.
