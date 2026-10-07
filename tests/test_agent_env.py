"""ADR-022 ClaimEnv: grounding is rewarded, fluency is not; abstention is honest only when justified."""
from berkelium.agent import AlwaysHighest, CheapestSufficient, Overconfident, Task, run_episode
from berkelium.evidence import Claim, Est, Law
from berkelium.evidence.library import point

P = point(L=(100, "mm"), h=(10, "mm"))


def tools(cheap_bound=None, fem_val=0.9):
    return [Law("cheap", "q", "mm", "analytic", 1.0, lambda p: Est(0.95, "mm", 0.0, cheap_bound)),
            Law("fem", "q", "mm", "numerical", 1000.0, lambda p: Est(fem_val, "mm", 0.01, 0.0))]


def task(truth="pass", **kw):
    return Task("t", Claim("q", "<=", 1.0, "mm"), P, tools(**kw), truth)


def test_ungrounded_verdict_is_punished_even_when_correct():
    ep = run_episode(task(), Overconfident())
    assert ep["outcome"] == "ungrounded" and ep["return"] == -1.0
    assert ep["trajectory"][-1]["info"]["correct_by_luck"] is True


def test_grounded_policies_and_cost_awareness():
    hi = run_episode(task(), AlwaysHighest())
    cs = run_episode(task(cheap_bound=0.02), CheapestSufficient())        # cheap tool decisive -> no FEM
    assert hi["outcome"] == cs["outcome"] == "correct" and cs["return"] > hi["return"] and cs["cost"] == 1.0
    cs2 = run_episode(task(cheap_bound=None), CheapestSufficient())       # unknown bound -> escalates
    assert cs2["outcome"] == "correct" and cs2["cost"] == 1001.0


def test_abstention_semantics_and_conflict():
    lazy = run_episode(task(cheap_bound=0.2, fem_val=0.9), _Abstain())
    assert lazy["outcome"] == "lazy_abstention" and lazy["return"] < 0
    conflict = run_episode(task(cheap_bound=0.001, fem_val=0.5), CheapestSufficient())
    assert conflict["outcome"] in ("correct", "honest_abstention")
    t = task(truth="insufficient_evidence", cheap_bound=None, fem_val=0.995)  # FEM interval straddles 1.0
    assert run_episode(t, CheapestSufficient())["outcome"] == "honest_abstention"


class _Abstain:
    name = "abstain"

    def act(self, obs):
        return {"type": "declare", "verdict": "insufficient_evidence"}
