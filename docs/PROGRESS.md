# Progress log (repository-native memory for future sessions)

Read this first, then docs/AGI_ENGINEERING.md, docs/IMPLEMENTATION_STATUS.md, docs/ARCHITECTURE_DECISIONS.md.

| Commit | What | Evidence file |
|---|---|---|
| 8a26f52 | M1 foundation: schemas, geometry, gear CEM, validation, dataset, gateway, training | tests |
| 19b9d5e..76d686f | Physics: FEM elasticity/conduction, Gmsh, GCI, L7, beam CEM | docs/IMPLEMENTATION_STATUS.md |
| f05d536 | ADR-019 evidence calculus | docs/experiments/evidence_e1.json |
| 7880695 | ADR-020 acausal law networks | docs/experiments/laws_e2.json |
| 1310852 | ADR-021 memory + gated discovery | docs/experiments/discovery_e3.json, memory_e3.jsonl |
| 76b526d, 21a0055 | ADR-022 ClaimEnv + leak fixes + LLM eval runner | docs/experiments/agent_e4.json |
| 24a2f00 + this commit | ADR-023 model synthesis + ModelEnv; ADR-024 generational training loop, DPO, curriculum | docs/experiments/model_e5.json |

| 4fd24fa, c56b37e (owner) | MI300X setup script; AGENT-V1/V2 datasets (byte-identical to rebuild) | out/agent*/manifest.json |
| HF (owner, 2026-10-07/08) | M1, AGENT-V1, AGENT-V2, AGENT-V2-DPO adapters; lineage verified | docs/manifests/model_generations.json |
| (this commit) | Endgame audit | docs/ENDGAME_AUDIT.md |

## Open items (in order) — superseded by docs/ENDGAME_AUDIT.md section 7
1. MI300X: run docs/MI300X_ENDGAME_RUNBOOK.md steps 0-7; commit eval JSONs + run manifests.
2. Interpret LLM results per generation; if V2/DPO do not beat BASE on held-out grounded_correct_rate without raising
   wrong_rate, record that as the result (do not tune the benchmark).
3. Relation library growth: fragments are human-authored; next is LLM-proposed fragments admitted only through the
   quarantine + discovery gate (ADR-021) — model-invented knowledge under deterministic control.
4. Geometry in the loop: ModelEnv actions that build Geometry IR whose measured quantities feed the law network.
