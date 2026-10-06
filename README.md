# BERKELIUM — core

AI-native computational engineering: natural-language intent → `DesignProposal` → deterministic
CEMs / procedural Geometry IR → geometry backends → measurement → validation → `DesignRecord`,
plus the verified dataset factory and Qwen3-32B training/evaluation loop that learns this workflow.

Architecture (source of truth): `docs/BERKELIUM_ARCHITECTURE.md` · decisions: `docs/ARCHITECTURE_DECISIONS.md` ·
sources & licenses: `docs/SOURCES_AND_LICENSES.md` · implementation status: `docs/IMPLEMENTATION_STATUS.md`.

**Authority rule.** A model may only *propose* (intent, requirements, CEM choice, parameters, procedural
geometry, patches). Measurements, engineering calculations and validation are written only by
deterministic code. Nothing here is "physically validated": no simulation adapters exist yet, and the
schema makes `physically_validated` permanently false until they do.

```bash
pip install -e ".[occt,mesh,dev]"
pytest
```

License: **undecided** (open decision U8). Do not accept external contributions until it is set.

Part of the AMD Development Hackathon.
