# AGI for engineering — operational definition, capability ladder, measured position

## The hardest question, answered as architecture
An engineering *assistant* maps a request to an answer with a fixed competence. An engineering *general
intelligence* must (1) construct the model it needs for an unfamiliar problem, (2) decide what evidence would
settle a question and acquire it, (3) refuse to decide when evidence is insufficient or contradictory, and
(4) **convert verified experience into new competence** — so that the cost or reach of tomorrow's work changes
because of today's — without a human re-coding it, and without any step that lets an unverified belief become
"knowledge". Berkelium's architecture therefore centres on four mechanisms, each now executable:

| Mechanism | Module | What it makes possible |
|---|---|---|
| Acausal law networks | `berkelium.laws` (ADR-020) | model construction by composition; causality, inversion, diagnostics derived, not coded |
| Evidence calculus | `berkelium.evidence` (ADR-019) | fidelity selection, escalation, CONFLICT / INSUFFICIENT_EVIDENCE as first-class outcomes |
| Engineering memory | `berkelium.memory` (ADR-021) | computable, content-addressed experience; nothing computed twice |
| Gated discovery | `berkelium.discovery` (ADR-021) | verified experience -> candidate law -> falsification (limits, held-out) -> promoted knowledge |

The language model is deliberately absent from that table: it is the planner/interface over these mechanisms,
never their authority.

## Capability ladder (each rung has an executable test; status = what evidence exists TODAY)
| Rung | Capability | Test | Status |
|---|---|---|---|
| E0 | solve known problems correctly | analytic + FEM verification suite (tests/test_physics.py) | **met** for linear elasticity, steady conduction, isentropic flow, pipe flow |
| E1 | construct a model from structured intent | law-network composition, E2 rocket chamber (2 new relations) | **partial**: composition works; relations are human-authored; LLM not yet in this loop |
| E2 | choose evidence/fidelity autonomously and stop when sufficient | E1 search; E3 benchmark | **met for scalar claims** (FEM 9 -> 2 per search; 10 -> 6 on fresh claims, 0 wrong verdicts) |
| E3 | transfer across domains | same engine on beam, vessel, nozzle, wall, pipe | **architectural transfer met** (no new engine code per domain); **learned transfer not measured** |
| E4 | learn from verified experience | E3: evidence -> promoted law -> cheaper future decisions | **demonstrated once** (one quantity, one family) |
| E5 | construct novel models | — | not started (needs LLM-proposed law networks + verification) |
| E6 | discover relations | E3 sparse discovery in automatically derived Pi groups, limit + held-out gate | **narrow**: monomial library, 1 quantity; discovered law is a box-limited surrogate, not a physical theory |
| E7 | investigate unfamiliar problems autonomously | — | not started |
| E8 | design-simulate-verify-improve novel systems with minimal decomposition | — | not started |

Generality, not rung count, is the objective; a rung is only "met" with the stated test passing.

## Biggest gap (next bottleneck)
The model is not yet an *agent inside* these mechanisms. Next: an agentic environment where Qwen proposes
relations / law networks / claims / evidence plans, the core verifies, and the trajectories (including
falsified proposals) become training data — so the learning loop includes the planner, not only the knowledge.
