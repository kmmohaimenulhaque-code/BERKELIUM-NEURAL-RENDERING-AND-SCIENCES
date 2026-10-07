"""LLM-policy plumbing, training-data builder invariants (no leakage, family split), endpoint-loss abort."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from berkelium.agent import GatewayPolicy, Task, messages, run_episode  # noqa: E402
from berkelium.ai.gateway import ReplayProvider  # noqa: E402
from berkelium.evidence import Claim, Est, Law  # noqa: E402
from berkelium.evidence.library import point  # noqa: E402


def _task():
    tools = [Law("cheap", "q", "mm", "analytic", 1.0, lambda p: Est(0.5, "mm", 0.0, 0.01)),
             Law("fem", "q", "mm", "numerical", 1000.0, lambda p: Est(0.5, "mm", 0.0, 0.0))]
    return Task("t", Claim("q", "<=", 1.0, "mm"), point(L=(100, "mm")), tools, "pass", family="heldout")


def test_gateway_policy_drives_env_and_bad_output_is_contained():
    good = ReplayProvider(['{"type":"evaluate","tool":"cheap"}', '{"type":"declare","verdict":"pass"}'])
    ep = run_episode(_task(), GatewayPolicy(good))
    assert ep["outcome"] == "correct" and ep["return"] > 0.99
    bad = ReplayProvider(["I think it passes.", "still prose", "more prose", "and more"])
    ep = run_episode(_task(), GatewayPolicy(bad))
    assert ep["outcome"] == "budget_exhausted" and ep["return"] == -1.0      # never crashes, never scores well


def test_prompt_format_is_deterministic():
    obs = {"b": 1, "a": {"y": 2, "x": 1}}
    assert messages(obs)[1]["content"] == '{"a": {"x": 1, "y": 2}, "b": 1}'


def test_training_builder_has_no_leakage_and_respects_family_split(tmp_path):
    import agent_build_training as B
    B.main(["--out", str(tmp_path)])
    rows = [json.loads(x) for x in (tmp_path / "train.chat.jsonl").read_text().splitlines()]
    rows += [json.loads(x) for x in (tmp_path / "val.chat.jsonl").read_text().splitlines()]
    assert rows and all("_nu0.2_" in r["id"] or "_nu0.4_" in r["id"] for r in rows)
    first = [r for r in rows if r["id"].endswith("#0")]
    assert first and all(json.loads(r["messages"][1]["content"])["evidence"] == [] for r in first)  # no future evidence
    pairs = [json.loads(x) for x in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert pairs and all('"declare"' in p["rejected"][0]["content"] and '"evaluate"' in p["chosen"][0]["content"]
                         for p in pairs)
    assert not {r["id"].split("#")[0] for r in rows} & {p["id"].split("#")[0] for p in pairs} - \
        {r["id"].split("#")[0] for r in rows}


def test_eval_aborts_when_endpoint_is_dead(tmp_path, monkeypatch):
    import agent_llm_eval as E
    monkeypatch.setenv("BERKELIUM_MODEL_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("BERKELIUM_MODEL", "Qwen/Qwen3-32B")
    out = tmp_path / "r.json"
    assert E.main(["--split", "heldout", "--label", "x", "--out", str(out)]) == 3 and not out.exists()
