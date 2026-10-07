import copy
import json

import jsonpatch

from berkelium.datasets.factory import FactoryConfig, build, load_split
from berkelium.pipeline import run


def test_factory_verified_deterministic_and_leak_free(tmp_path):
    cfg = FactoryConfig(seed=3, n_valid=6, n_boundary=2, n_invalid=3, n_procedural=1, realize=False)
    m1 = build(tmp_path / "a", cfg)
    m2 = build(tmp_path / "b", cfg)
    assert m1["manifest_hash"] == m2["manifest_hash"] and m1["files"] == m2["files"]
    assert m1["family_leakage"] == []
    rows = [r for f in m1["files"] for r in load_split(tmp_path / "a" / f)]
    assert rows and all(r["verification"]["passed"] for r in rows)
    for r in rows:
        assert {"id", "schema_version", "task", "messages", "target", "verification", "source", "cem_refs",
                "family_id", "split"} <= set(r)
        if r["task"] == "plan":
            # the plan target never contains evaluation data and re-verifies
            assert "evaluation" not in json.dumps(r["target"])
            assert run(r["target"], realize=False).record.evaluation.validation.summary in ("pass", "warn")
        if r["task"] == "repair":
            broken = json.loads(r["messages"][1]["content"].split("PROPOSAL:\n")[1].split("\nVALIDATION:")[0])
            fixed = jsonpatch.apply_patch(copy.deepcopy(broken), r["target"])
            assert run(broken, realize=False).record.evaluation.validation.summary == "fail"
            assert run(fixed, realize=False).record.evaluation.validation.summary in ("pass", "warn")
    assert all(r["split"] == "heldout_family" for r in rows if r["task"] == "plan_procedural")
