# Sources and Licenses Ledger

Every external project studied or used by Berkelium. **No third-party code has been copied into Berkelium.** Update this file before any reuse.

Status key — **Studied**: read for architecture/API understanding only. **Dependency**: used as a library/process, unmodified. **Reused**: source copied or adapted (requires attribution below).

## Inspected repositories (2026-10-05)

| Project | URL | Commit | License | Relevant component | Purpose | Status |
|---|---|---|---|---|---|---|
| PicoGK | github.com/leap71/PicoGK | 0e6cf6b | Apache-2.0 | `Voxels`, `Lattice`, `IImplicit`, `Mesh`, IO (STL/CLI/VDB) | Voxel/implicit kernel API study; candidate implicit backend | Studied |
| PicoGKRuntime | github.com/leap71/PicoGKRuntime | 0f26321 | Apache-2.0 (submodules: OpenVDB, GLFW, imgui) | C ABI `API/PicoGK.h`, CMake build | Assess Linux build feasibility | Studied |
| LEAP71_ShapeKernel | github.com/leap71/LEAP71_ShapeKernel | 7b90978 | Apache-2.0 | `BaseShape`, `LocalFrame`, `SurfaceModulation`/`LineModulation`, splines, lattice pipes | Architectural inspiration for frame-parameterised, modulated shapes | Studied |
| LEAP71_RoverWheel | github.com/leap71/LEAP71_RoverWheel | 6cfe3cd | Apache-2.0 | `RoverWheel` base, layer conformal mapping, tread patterns | CEM structure study | Studied |
| LEAP71_HelixHeatX | github.com/leap71/LEAP71_HelixHeatX | a0a0234 | Apache-2.0 | Two-fluid helical voids, wall-offset separation, print supports | Multi-domain CEM study | Studied |
| berkelium-studio | github.com/sudeep03042008/berkelium-studio | ff7177c | **None declared** | FastAPI backend, Electron/React shell | Team code (Studio) | Internal |
| BERKELIUM-NEURAL-RENDERING-AND-SCIENCES | github.com/kmmohaimenulhaque-code/… | 50c1fa3 | **None declared** | Directory skeleton | Team code (ML) | Internal |

## Libraries evaluated in this session

| Project | Version tested | License | Purpose | Status | Notes |
|---|---|---|---|---|---|
| cadquery-ocp (OCP, Python bindings to OCCT) | 8.0.1.0.0 | Bindings Apache-2.0; **OCCT itself LGPL-2.1 + OCCT exception** | Exact BREP backend, STEP export | Dependency (proposed) | Smoke-tested on Linux: boolean, `BRepCheck`, STEP write OK. Dynamic linking satisfies LGPL; do not statically link a modified OCCT without complying. |
| manifold3d | 3.5.4 | Apache-2.0 (verify in package — metadata field empty) | Robust mesh booleans, level-set meshing, preview | Dependency (proposed) | Smoke-tested: boolean + gyroid level set (214k tris, 0.43 s, CPU). |
| Qwen3-32B | — | Apache-2.0 (verify model card at download time) | Primary LLM | Dependency (proposed) | Weights not yet downloaded. |

## Candidates recorded, not yet evaluated

| Project | License | Concern |
|---|---|---|
| OpenVDB | Apache-2.0 (relicensed from MPL-2.0 ~2020; **verify**) | Native build complexity |
| FreeCAD | LGPL-2.0+ | Application, not an embeddable kernel; interop only |
| CalculiX (FEA) | GPL-2.0 | Invoke as separate process only; never link |
| OpenFOAM (CFD) | GPL-3.0 | Separate process only |
| FEniCSx | LGPL-3.0 | Acceptable as library |
| vLLM (ROCm) | Apache-2.0 | ROCm version pinning |
| HF Transformers / PEFT / TRL | Apache-2.0 | — |

## Inspiration vs. reuse — LEAP 71

- **Architectural inspiration (adopted):** CEM = code that encodes engineering logic and emits geometry; local-frame parameterisation with normalised ratios; modulation of dimensions over a shape's parameter space; robust implicit/voxel booleans for AM-oriented internals; constructing separation by offsetting one domain from another (HelixHeatX).
- **API/reference study (not adopted):** PicoGK's handle-based C ABI; ShapeKernel's `vox*/msh*` naming.
- **Deliberately rejected patterns:** static mutable parameters (RoverWheel `m_fHubRadius` etc. are `static`), viewer side-effects inside construction (`Sh.PreviewVoxels`, screenshots in `voxConstruct`), hard-coded dimensions per subclass instead of typed parameter schemas, modulations as non-serialisable delegates.
- **Permissible code reuse:** Apache-2.0 allows it with license copy, NOTICE retention (no NOTICE files present in these repos as of the inspected commits), and statement of changes. **None performed.**
- **Independent implementation:** all Berkelium code is written from scratch in Python.

## Action items
1. Both team repos need a LICENSE before external contributors or any reuse decisions.
2. Re-verify manifold3d, OpenVDB and Qwen3 licenses from primary sources when they become actual dependencies.
