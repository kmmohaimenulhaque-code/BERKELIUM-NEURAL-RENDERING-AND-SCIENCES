import json

from fastapi.testclient import TestClient

from berkelium.api.app import create_app
from berkelium.cli import main
from berkelium.datasets.factory import FactoryConfig, build
from berkelium.training.data import length_audit, render, to_chat

GEAR = {"structure": {"components": [{"id": "pair", "kind": "cem", "cem": "gear.spur_pair@0.1",
                                      "parameters": {"module": 2, "z1": 20, "z2": 40, "face_width": 25}}]}}


def test_api_design_lifecycle(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    assert c.get("/v1/schema/DesignProposal").json()["$schema"].endswith("2020-12/schema")
    assert c.get("/v1/cems").json()[0]["ref"] == "gear.spur_pair@0.1"
    r = c.post("/v1/designs?backend=manifold", json=GEAR).json()
    did = r["id"]
    assert r["record"]["evaluation"]["validation"]["summary"] in ("pass", "warn")
    glb = next(a for a in r["record"]["evaluation"]["artifacts"] if a["format"] == "glb")
    assert c.get(f"/v1/artifacts/{glb['sha256']}.glb").status_code == 200
    p = c.patch(f"/v1/designs/{did}?realize=false",
                json=[{"op": "replace", "path": "/structure/components/0/parameters/z1", "value": 12}]).json()
    assert p["revision"] == 1 and p["record"]["evaluation"]["validation"]["summary"] == "fail"
    assert c.get(f"/v1/designs/{did}/validation").json()["summary"] == "fail"
    assert c.post("/v1/designs", json={"structure": {}}).status_code == 422
    assert c.post("/v1/intents", json={"intent": "x"}).status_code == 503   # no model configured


def test_sft_render_refuses_unverified_and_audits(tmp_path):
    build(tmp_path / "ds", FactoryConfig(seed=5, n_valid=4, n_boundary=1, n_invalid=2, n_procedural=1, realize=False))
    info = render([tmp_path / "ds" / "train.jsonl"], tmp_path / "train.chat.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "train.chat.jsonl").read_text().splitlines()]
    assert info["rows"] == len(rows) > 0 and rows[0]["messages"][-1]["role"] == "assistant"
    assert length_audit(tmp_path / "train.chat.jsonl")["n"] == len(rows)
    import pytest
    with pytest.raises(ValueError):
        to_chat({"id": "x", "verification": {"passed": False}})


def test_cli_run_and_schemas(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps(GEAR))
    assert main(["run", str(f), "--no-realize"]) == 0
    assert main(["schemas", "--out", str(tmp_path / "schema")]) == 0
