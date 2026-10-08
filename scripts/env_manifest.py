"""Reproducibility manifest of the CURRENT machine (Phase 1). Never installs anything.
  python scripts/env_manifest.py > out/env_manifest.json"""
import importlib
import json
import platform
import shutil
import subprocess
import sys


def ver(m):
    try:
        mod = importlib.import_module(m)
        return getattr(mod, "__version__", "installed")
    except Exception:  # noqa: BLE001
        return None


def cmd(c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=30).stdout.strip()[:400] or None
    except Exception:  # noqa: BLE001
        return None


m = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
     "packages": {p: ver(p) for p in ("torch", "transformers", "peft", "accelerate", "datasets", "trl", "vllm",
                                      "numpy", "scipy", "sympy", "skfem", "meshio", "pydantic", "OCP", "manifold3d")},
     "git_head": cmd(["git", "rev-parse", "HEAD"]), "git_dirty": bool(cmd(["git", "status", "--porcelain"])),
     "executables": {e: shutil.which(e) for e in ("gmsh", "ccx", "vllm", "rocm-smi", "rocminfo")}}
try:
    import torch
    m["torch_hip"] = getattr(torch.version, "hip", None)
    m["gpu"] = [{"name": torch.cuda.get_device_name(i), "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                 "arch": getattr(torch.cuda.get_device_properties(i), "gcnArchName", None)}
                for i in range(torch.cuda.device_count())]
except Exception:  # noqa: BLE001
    m["gpu"] = []
m["gmsh_version"] = cmd(["gmsh", "--version"]) if m["executables"]["gmsh"] else None
print(json.dumps(m, indent=2))
