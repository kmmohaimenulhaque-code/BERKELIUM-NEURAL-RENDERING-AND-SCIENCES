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

## Engineering formula sources (implemented in `berkelium/cem/library/gear/formulas.py`)

Formulas are re-implemented from these references; no text or code is copied. Numeric oracle tests are in
`tests/test_gear_formulas.py`.

| Source | Used for | Status / notes |
|---|---|---|
| KHK Gear Technical Reference (KHK Co.) | inv α (Eq. 3.2), z_min = 2/sin²α (Eq. 3.9), x_min (Eq. 3.10), working pressure angle, centre distance, y, inverse calculation (Table 4.4), transverse contact ratio | Studied; formulas re-implemented. Oracle: m3, z 12/24, x 0.6/0.36 → α_w 26.0886°, a 56.4999 mm |
| Budynas & Nisbett, *Shigley's Mechanical Engineering Design*, 8th ed., McGraw-Hill | Eqs. 13-10…13-13 (interference limits), 14-4…14-8 (Barth K_v, Lewis), Table 14-2 (Lewis Y, 20° FD, 25 values re-keyed with citation), §14-1 (face width 3p–5p, CR ≈ 1.5), basic rack 1.0/1.25/0.300 m | Studied. Lewis = preliminary only; Y table used only for x = 0, 12–400 teeth, else `not_evaluated` |
| ANSI/AGMA 2001-D04 | K_v ≥ 1 convention only | Rating NOT implemented → `not_evaluated` |
| ISO 53 / 21771 / 6336 | — | Not purchased; no ISO values used |
| Line-of-action / rack-generation geometry | involute interference margin, fillet-contact check, root clearance, trochoid fillet by rack simulation | Derived (no constants); verified numerically (exact involute to 1e-12) |

Unverified/unsourced thresholds deliberately NOT used: contact ratio ≥1.2/1.4, "practical" 14-tooth undercut limit,
ISO 0.38 m rack tip radius, backlash formulas (backlash is a designer-chosen thinning parameter).

## Dependencies now in use

| Package | Version tested | License | Use |
|---|---|---|---|
| pydantic | 2.13 | MIT | schemas |
| rfc8785 | 0.1.4 | Apache-2.0 | canonical JSON hashing |
| jsonpatch | 1.33 | BSD-3-Clause | repair patches |
| cadquery-ocp(-novtk) | 8.0.1.1 (7.9 compatible) | Apache-2.0 bindings; OCCT LGPL-2.1 + exception (dynamic) | exact BREP, STEP |
| manifold3d | 3.5.4 | Apache-2.0 | mesh, level sets |
| trimesh | 5.1 | MIT | STL/GLB export |
| numpy | 2.4 | BSD-3-Clause | numerics |

## Physics stack (PHYSICS milestone, checked 2026-10-07)
| Project | Version used/checked | License | Use | Reused or studied | Notes |
|---|---|---|---|---|---|
| scikit-fem | 12.0.2 (PyPI) | BSD-3-Clause | in-process FEM assembly | dependency (`[physics]` extra) | cite Gustafsson & McBain 2020, JOSS 5(52):2369 |
| meshio | 5.3.5 | MIT | reads Gmsh .msh, writes VTU | dependency | |
| SciPy | as installed | BSD-3-Clause | sparse direct / CG solves | dependency | |
| Gmsh | 4.15.2 (executable) | GPL-2.0-or-later (with linking exceptions) | STEP -> tet mesh | **external process only**; never imported | ADR-016 |
| CalculiX ccx | 2.21 (Debian/Ubuntu package) | GPL-2.0 | planned cross-check solver | not used yet; external process only | |
| OpenFOAM | not installed / version not checked | GPL-3.0 | planned CFD | adapter detects executables only | |
| FEniCSx (DOLFINx) | 0.11 (June 2026, fenicsproject.org) | LGPL-3.0 (DOLFINx), MIT (Basix) | candidate large-scale FEM | studied only | conda/apt/Docker install |

### Formula / method references used by deterministic code
Celik et al. 2008, J. Fluids Eng. 130(7):078001 (GCI procedure) · Roache 1994, J. Fluids Eng. 116:405-413 ·
Colebrook 1939, J. Inst. Civil Eng. 11:133-156 · White, Fluid Mechanics (Darcy-Weisbach, Moody accuracy) ·
Gere & Goodno, Mechanics of Materials (cantilever, flexure) · Cowper 1966 (shear coefficient, tests only) ·
Budynas & Nisbett, Shigley's MED Table A-5 and Bergman et al., Fundamentals of Heat and Mass Transfer Table A.1
(material library; nominal class values).
