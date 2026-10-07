"""Core API v1. Studio consumes schemas, CEM parameter schemas, records, validation and artifacts from here.
Studio never calls the LLM and never computes engineering (ADR-010). In-memory design store for now."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import jsonpatch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .. import __version__
from ..cem.protocol import default_registry
from ..pipeline import ArtifactStore, run
from ..schema.export import EXPORTED, json_schema

ARTIFACT_EXT = {"step": "step", "glb": "glb", "stl": "stl"}


def create_app(artifact_root: str | None = None) -> FastAPI:
    app = FastAPI(title="Berkelium Core", version=__version__)
    store = ArtifactStore(artifact_root or os.environ.get("BERKELIUM_ARTIFACTS", "artifacts"))
    designs: dict[str, list[dict]] = {}   # id -> revisions (proposal + record)

    def _execute(proposal: dict, realize: bool, backend: str | None):
        res = run(proposal, realize=realize, backend=backend, artifacts=store if realize else None)
        if res.record is None:
            raise HTTPException(422, {"schema_errors": res.schema_errors})
        return res

    @app.get("/v1/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/v1/schema/{name}")
    def schema(name: str):
        if name not in EXPORTED:
            raise HTTPException(404, f"unknown schema; available: {sorted(EXPORTED)}")
        return json_schema(name)

    @app.get("/v1/cems")
    def cems():
        return [{k: v for k, v in c.items() if k not in ("requirements_schema", "parameters_schema")}
                for c in default_registry().summary()]

    @app.get("/v1/cems/{ref}")
    def cem(ref: str):
        for c in default_registry().summary():
            if c["ref"] == ref:
                return c
        raise HTTPException(404, "unknown CEM")

    @app.post("/v1/designs")
    def create(proposal: dict, realize: bool = True, backend: str | None = None):
        res = _execute(proposal, realize, backend)
        rec = res.record
        designs.setdefault(rec.id, []).append({"proposal": proposal, "record": rec.model_dump(mode="json")})
        return {"id": rec.id, "revision": len(designs[rec.id]) - 1, "record": rec.model_dump(mode="json"),
                "job": [s.__dict__ for s in res.job.stages]}

    @app.get("/v1/designs/{did}")
    def get(did: str, rev: int | None = None):
        revs = designs.get(did) or (_ for _ in ()).throw(HTTPException(404, "unknown design"))
        return revs[-1 if rev is None else rev]["record"]

    @app.patch("/v1/designs/{did}")
    def patch(did: str, ops: list[dict], realize: bool = True, backend: str | None = None):
        """User edits and model repairs share this path: JSON Patch against the latest proposal."""
        if did not in designs:
            raise HTTPException(404, "unknown design")
        base = designs[did][-1]["proposal"]
        try:
            new = jsonpatch.apply_patch(copy.deepcopy(base), ops)
        except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as e:
            raise HTTPException(422, str(e)) from None
        res = _execute(new, realize, backend)
        rec = res.record.model_copy(update={"revision": len(designs[did])})
        designs[did].append({"proposal": new, "record": rec.model_dump(mode="json")})
        return {"id": did, "revision": rec.revision, "record": rec.model_dump(mode="json")}

    @app.get("/v1/designs/{did}/validation")
    def validation(did: str):
        return get(did)["evaluation"]["validation"]

    @app.get("/v1/artifacts/{sha}.{ext}")
    def artifact(sha: str, ext: str):
        if ext not in ARTIFACT_EXT or len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
            raise HTTPException(400, "bad artifact reference")
        p: Path = store.path(sha, ext)
        if not p.exists():
            raise HTTPException(404, "artifact not found")
        return FileResponse(p, media_type="model/gltf-binary" if ext == "glb" else "application/octet-stream")

    @app.post("/v1/intents")
    def intent(body: dict):
        """Natural language -> model proposal -> core -> repair loop. Needs BERKELIUM_MODEL_URL/BERKELIUM_MODEL."""
        from ..ai.gateway import GatewayError, provider_from_env
        from ..ai.orchestrator import Orchestrator
        try:
            prov = provider_from_env()
        except GatewayError as e:
            raise HTTPException(503, str(e)) from None
        s = Orchestrator(prov).design(body["intent"], max_repairs=int(body.get("max_repairs", 2)))
        return {"attempts": [{k: v for k, v in a.__dict__.items() if k != "raw"} for a in s.attempts],
                "proposal": s.proposal, "record": s.record.model_dump(mode="json") if s.record else None}

    return app
