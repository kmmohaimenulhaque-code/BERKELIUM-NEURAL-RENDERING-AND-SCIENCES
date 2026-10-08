"""ADR-023/024: model synthesis, ModelEnv judging, data builders, LLM-eval plumbing, failure analysis."""
import json
import sys

import pytest

from berkelium.agent.model_env import ModelEnv, ModelTask, run_model_episode
from berkelium.agent.model_policies import ConstructorPolicy, GreedyRetrieval
from berkelium.agent.model_tasks import TRAIN_DOMAINS, tasks
from berkelium.synthesis import Problem, construct, load
from berkelium.units import Quantity as Q

CANT = frozenset({"support:cantilever", "load:end", "section:rect"})
SS = frozenset({"support:simply_supported", "load:mid", "section:rect"})
VES = frozenset({"vessel:cylinder", "load:internal_pressure", "end:plane_strain"})
PIPE = frozenset({"flow:pipe", "flow:fully_developed"})
BEAM = {"P": Q.of(200, "N"), "L": Q.of(100, "mm"), "E": Q.of(207, "GPa"), "b": Q.of(5, "mm"), "h": Q.of(10, "mm")}


def test_library_quarantines_dimensionally_invalid_fragment():
    ok, quarantined = load()
    assert [q for q, _ in quarantined] == ["BROKEN_deflection_units"] and len(ok) == 30


def test_context_selects_the_right_model():
    inertia = 5 * 10 ** 3 / 12 * 1e-12
    c = construct(Problem(CANT, BEAM, "delta"))
    s = construct(Problem(SS, BEAM, "delta"))
    assert c.status == s.status == "determined"
    assert abs(c.value - 200 * 0.1 ** 3 / (3 * 207e9 * inertia) * 1e3) < 1e-9
    assert abs(s.value - 200 * 0.1 ** 3 / (48 * 207e9 * inertia) * 1e3) < 1e-9


def test_validity_competition_contradiction_ambiguity_and_gaps():
    thick = construct(Problem(VES, {"p": Q.of(10, "MPa"), "ri": Q.of(10, "mm"), "ro": Q.of(20, "mm"), "nu": Q.of(0.3)}, "svm"))
    assert thick.status == "determined" and "thin_wall_hoop" not in thick.chosen and abs(thick.value - 23.1325) < 1e-3
    thin = construct(Problem(VES, {"p": Q.of(1, "MPa"), "ri": Q.of(100, "mm"), "ro": Q.of(102, "mm")}, "sig_t"))
    assert thin.status == "determined" and len([c for c in thin.candidates if c.status == "solved"]) == 2
    contra = construct(Problem(VES, {"p": Q.of(10, "MPa"), "ri": Q.of(10, "mm"), "ro": Q.of(20, "mm"),
                                     "w": Q.of(12, "mm")}, "sig_t"))
    assert contra.status == "contradictory"
    amb = construct(Problem(frozenset({"flow:isentropic"}), {"gamma": Q.of(1.4), "eps": Q.of(1.6875)}, "M_e"))
    assert amb.status == "ambiguous"
    gap = construct(Problem(CANT, {k: v for k, v in BEAM.items() if k != "h"}, "delta"))
    assert gap.status == "underdetermined" and "h" in gap.missing
    pipe = {"rho_f": Q.of(998, "kg/m^3"), "mu": Q.of(1.002e-3, "Pa*s"), "D": Q.of(50, "mm"), "Lp": Q.of(10, "m"),
            "rough": Q.of(0, "mm"), "Vf": Q.of(0.06, "m/s")}                     # Re ~ 3000: transitional
    assert construct(Problem(PIPE, pipe, "dp")).status == "outside_validity"


def _task(ctx, known, target, status, value=None, domain="structures"):
    return ModelTask("t#0", domain, ctx, known, target, status, value, "train")


def test_modelenv_requires_grounding_and_punishes_wrong_context():
    truth = construct(Problem(SS, BEAM, "delta")).value
    t = _task(SS, BEAM, "delta", "determined", truth)
    env = ModelEnv(t)
    env.step({"type": "search", "var": "delta"})
    _, r, _, info = env.step({"type": "declare", "status": "determined", "value": truth})
    assert info["outcome"] == "ungrounded" and r == -1.0                      # right number, no own solve
    assert run_model_episode(t, ConstructorPolicy())["outcome"] == "correct"
    g = run_model_episode(t, GreedyRetrieval())                               # picks the cantilever formula
    assert g["outcome"] == "wrong"
    assert g["trajectory"][-1]["obs"]["last_solve"]["context_violations"]


def test_judge_closure_respects_validity_domains():
    k = {"rho_f": Q.of(998, "kg/m^3"), "mu": Q.of(1.002e-3, "Pa*s"), "D": Q.of(50, "mm"), "Lp": Q.of(10, "m"),
         "rough": Q.of(0.01, "mm"), "Vf": Q.of(1.0, "m/s")}
    c = construct(Problem(PIPE, k, "dp"))
    ep = run_model_episode(_task(PIPE, k, "dp", "determined", c.value, "fluids"), ConstructorPolicy())
    assert ep["outcome"] == "correct"
    assert ep["trajectory"][-1]["obs"]["last_solve"]["closure_violations"] == []   # laminar f not enforced at Re~5e4


def test_benchmark_is_reproducible_and_split_by_domain():
    a, b = tasks(2, seed=0), tasks(2, seed=0)
    assert [(t.id, t.truth_status, t.truth_value) for t in a] == [(t.id, t.truth_status, t.truth_value) for t in b]
    assert {t.split for t in a if t.domain in TRAIN_DOMAINS} == {"train"}
    assert {t.split for t in a if t.domain not in TRAIN_DOMAINS} == {"heldout"}


def test_builder_excludes_heldout_and_never_splits_episodes(tmp_path):
    sys.path.insert(0, "scripts")
    import model_build_training
    model_build_training.main(["--out", str(tmp_path), "--n-per", "6", "--seed", "7"])
    rows = [json.loads(x) for x in (tmp_path / "train.chat.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (tmp_path / "val.chat.jsonl").read_text().splitlines()]
    assert rows and val and all(r["domain"] in TRAIN_DOMAINS for r in rows + val)
    ep = lambda r: "#".join(r["id"].split("#")[:2])  # noqa: E731
    assert not ({ep(r) for r in rows} & {ep(r) for r in val})


class _ReplayConstructor:
    """Fake OpenAI-compatible provider that answers with the constructor's action (tests the plumbing only)."""
    model = "replay-constructor"

    def __init__(self):
        self.pol, self.task = None, None

    def check(self, wait_s=0.0):
        return [self.model]

    def generate(self, messages, schema=None, decoding=None):
        obs = json.loads(messages[1]["content"])
        if self.pol is None or obs["model"] == [] and obs["last_solve"] == {} and not obs["last_search"]:
            self.pol = ConstructorPolicy()
        return type("G", (), {"text": json.dumps(self.pol.act(obs))})()


def test_llm_eval_plumbing_with_replay_provider(tmp_path):
    sys.path.insert(0, "scripts")
    import model_llm_eval
    out = tmp_path / "r.json"
    assert model_llm_eval.main(["--split", "heldout", "--label", "replay", "--out", str(out), "--limit", "6"],
                               provider=_ReplayConstructor()) == 0
    r = json.loads(out.read_text())
    assert r["llm"]["grounded_correct_rate"] == 1.0 and r["llm"]["invalid_action_rate"] == 0.0


def test_failure_analysis_clusters_mechanisms(tmp_path):
    sys.path.insert(0, "scripts")
    import failure_analysis
    out = tmp_path / "c.json"
    failure_analysis.main(["docs/experiments/trajectories_e5.jsonl", "--policy", "greedy_retrieval", "--out", str(out)])
    spec = json.loads(out.read_text())
    assert spec["failures"] == 40 and "context_violation" in spec["clusters"]
    assert spec["oversample"]["beam_simply_supported_deflection"] > spec["oversample"]["beam_cantilever_deflection"]


def test_dpo_logprob_masks_prompt_tokens():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, "training")
    from dpo_lora import seq_logp

    class M(torch.nn.Module):
        def forward(self, input_ids):
            return type("O", (), {"logits": torch.zeros(1, input_ids.shape[1], 5)})()
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    assert abs(float(seq_logp(M(), ids, labels)) - 2 * torch.log(torch.tensor(0.2)).item()) < 1e-6
