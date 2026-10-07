import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import jsonpatch

from berkelium.ai.evaluate import evaluate
from berkelium.ai.gateway import OpenAICompatibleProvider, ReplayProvider
from berkelium.ai.orchestrator import Orchestrator, extract_json
from berkelium.datasets.factory import params_proposal, plan_target
from berkelium.cem.library.gear.cem import Parameters

GOOD = plan_target({"ratio": 2.0, "module": {"value": 2.0, "unit": "mm"}})
BROKEN = params_proposal(Parameters(module=2, z1=12, z2=24, face_width=25))
FIX = jsonpatch.make_patch(BROKEN, params_proposal(Parameters(module=2, z1=12, z2=24, x1=0.3, face_width=25))).patch


def test_extract_json_tolerates_fences_and_thinking():
    assert extract_json("<think>hmm {x}</think>```json\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json("Sure: [1, 2]") == [1, 2]


def test_plan_passes_first_try():
    s = Orchestrator(ReplayProvider([json.dumps(GOOD)])).design("2:1 gears, module 2")
    assert s.final_summary in ("pass", "warn") and len(s.attempts) == 1


def test_repair_loop_fixes_undercut():
    p = ReplayProvider([json.dumps(BROKEN), json.dumps(FIX)])
    s = Orchestrator(p).design("12-tooth pinion, 2:1")
    assert [a.kind for a in s.attempts] == ["plan", "repair"]
    assert s.attempts[0].summary == "fail" and s.final_summary in ("pass", "warn")
    assert "undercut" in p.calls[1][1]["content"]           # repair prompt carries validator evidence


def test_model_cannot_inject_evaluation():
    evil = dict(GOOD, evaluation={"validation": {"summary": "pass"}})
    s = Orchestrator(ReplayProvider([json.dumps(evil)])).design("x", max_repairs=0)
    a = s.attempts[0]
    assert a.parsed and not a.schema_valid and a.hallucinated_evaluation and s.record is None


def test_evaluate_metrics_against_ground_truth():
    ex = {"id": "e1", "split": "test", "task": "plan", "family_id": "f", "target": GOOD,
          "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "2:1, module 2"}]}
    wrong = plan_target({"ratio": 3.0, "module": {"value": 2.0, "unit": "mm"}})  # passes its OWN spec
    rep = evaluate([ex, dict(ex, id="e2")], Orchestrator(ReplayProvider([json.dumps(GOOD), json.dumps(wrong)])),
                   max_repairs=0)
    m = rep["metrics"]
    assert m["validation_pass_first"] == 1.0 and m["requirement_satisfaction"] == 0.5


def test_openai_compatible_payload_roundtrip():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"model": "qwen", "choices": [{"message": {"content": json.dumps(GOOD)}}],
                               "usage": {"completion_tokens": 9}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    prov = OpenAICompatibleProvider(f"http://127.0.0.1:{srv.server_port}/v1", "Qwen/Qwen3-32B")
    s = Orchestrator(prov).design("2:1, module 2")
    srv.shutdown()
    assert s.final_summary in ("pass", "warn")
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False} and seen["model"] == "Qwen/Qwen3-32B"
