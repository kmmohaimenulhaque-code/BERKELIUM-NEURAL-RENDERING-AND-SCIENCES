"""berkelium CLI: schemas | run | dataset | sft-render | model-check | eval | design | api"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="berkelium")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schemas"); s.add_argument("--out", default="docs/schema")
    s = sub.add_parser("run"); s.add_argument("proposal"); s.add_argument("--backend"); s.add_argument("--artifacts")
    s.add_argument("--no-realize", action="store_true"); s.add_argument("--out")
    s = sub.add_parser("dataset"); s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=0)
    for k, d in (("valid", 40), ("boundary", 10), ("invalid", 20), ("procedural", 6)):
        s.add_argument(f"--n-{k}", type=int, default=d)
    s.add_argument("--no-realize", action="store_true")
    s = sub.add_parser("sft-render"); s.add_argument("--dataset", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("model-check", help="verify BERKELIUM_MODEL_URL is up and serves BERKELIUM_MODEL")
    s.add_argument("--wait", type=float, default=0.0, help="seconds to keep polling (vLLM load takes minutes)")
    s.add_argument("--pidfile", default="out/logs/vllm.pid", help="abort at once if this server process dies")
    s.add_argument("--log", default=None, help="server log to show while waiting / on failure")
    s = sub.add_parser("eval"); s.add_argument("--dataset", required=True)
    s.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the endpoint before starting")
    s.add_argument("--splits", default="test,heldout_family"); s.add_argument("--max-repairs", type=int, default=2)
    s.add_argument("--unconstrained", action="store_true"); s.add_argument("--label", default="baseline")
    s.add_argument("--out", required=True)
    s = sub.add_parser("design"); s.add_argument("intent"); s.add_argument("--max-repairs", type=int, default=2)
    s = sub.add_parser("api"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8100)
    a = ap.parse_args(argv)

    if a.cmd == "schemas":
        from .schema.export import export_all
        for p in export_all(a.out):
            print(p)
    elif a.cmd == "run":
        from .pipeline import ArtifactStore, run
        res = run(Path(a.proposal).read_text(), realize=not a.no_realize, backend=a.backend,
                  artifacts=ArtifactStore(a.artifacts) if a.artifacts else None)
        if res.record is None:
            print(json.dumps(res.schema_errors, indent=2)); return 2
        v = res.record.evaluation.validation
        for r in v.results:
            print(f"L{r.level} {r.status:13} {r.target:22} {r.message}")
        print(f"summary={v.summary} counts={v.counts} hash={res.record.content_hash}")
        if a.out:
            Path(a.out).write_text(res.record.model_dump_json(indent=2))
        return 0 if v.summary in ("pass", "warn") else 1
    elif a.cmd == "dataset":
        from .datasets.factory import FactoryConfig, build
        m = build(a.out, FactoryConfig(seed=a.seed, n_valid=a.n_valid, n_boundary=a.n_boundary,
                                       n_invalid=a.n_invalid, n_procedural=a.n_procedural, realize=not a.no_realize))
        print(json.dumps({k: m[k] for k in ("files", "rejected", "counts", "family_leakage", "manifest_hash")}, indent=2))
    elif a.cmd == "sft-render":
        from .training.data import length_audit, render
        for split in ("train", "val", "test"):
            f = Path(a.dataset) / f"{split}.jsonl"
            if f.exists():
                info = render([f], Path(a.out) / f"{split}.chat.jsonl")
                print(split, info, length_audit(info["file"]))
    elif a.cmd in ("model-check", "eval", "design"):
        from .ai.gateway import GatewayError, provider_from_env
        from .ai.orchestrator import Orchestrator
        try:
            prov = provider_from_env()
            if a.cmd != "design":
                waiter = None
                if getattr(a, "wait", 0.0):
                    from .ai.waiting import make_waiter
                    waiter = make_waiter(prov.base_url, getattr(a, "pidfile", None), getattr(a, "log", None))
                ids = prov.check(wait_s=getattr(a, "wait", 0.0), on_wait=waiter)
                print(f"endpoint ok: {prov.base_url} serves {ids}", file=sys.stderr)
        except GatewayError as e:
            print(f"model endpoint not ready: {e}", file=sys.stderr)
            return 3
        if a.cmd == "model-check":
            return 0
        orch = Orchestrator(prov, constrained=not getattr(a, "unconstrained", False))
        if a.cmd == "design":
            s_ = orch.design(a.intent, max_repairs=a.max_repairs)
            print(json.dumps({"attempts": [{k: v for k, v in x.__dict__.items() if k != "raw"} for x in s_.attempts],
                              "proposal": s_.proposal}, indent=2))
            return 0
        from .ai.evaluate import EndpointLost, evaluate
        from .datasets.factory import load_split
        ex = [r for sp in a.splits.split(",") if (Path(a.dataset) / f"{sp}.jsonl").exists()
              for r in load_split(Path(a.dataset) / f"{sp}.jsonl")]
        try:
            rep = evaluate(ex, orch, max_repairs=a.max_repairs, out=a.out, label=a.label)
        except EndpointLost as e:
            print(f"evaluation aborted: {e}", file=sys.stderr)
            return 3
        print(json.dumps(rep["metrics"], indent=2))
    elif a.cmd == "api":
        import uvicorn

        from .api.app import create_app
        uvicorn.run(create_app(), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
