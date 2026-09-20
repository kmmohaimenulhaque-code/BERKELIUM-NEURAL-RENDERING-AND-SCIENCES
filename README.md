Part of AMD Development Hackathon 

# BERKELIUM — Neural Rendering & Sciences

> An experimental research and engineering repository for AI, neural rendering, 3D vision, scientific computing, and AMD GPU-accelerated workloads.

**Status:** 🚧 Early-stage research / infrastructure skeleton

---

## Overview

**Berkelium** is a modular research repository designed to bring together the tooling, datasets, experiments, models, and infrastructure required for advanced AI and scientific computing workloads.

The project focuses on:

- 🤖 AI / machine learning
- 🧠 Large language and vision models
- 👁️ OCR and computer vision
- 🎨 Neural rendering
- 🧊 3D geometry and reconstruction
- 🛰️ Scientific and space-related computing
- 📦 Large-scale multimodal datasets
- ⚡ AMD GPU / ROCm acceleration
- 🧪 Reproducible experiments and benchmarking

The repository currently provides the architectural foundation. Individual components, datasets, models, benchmarks, and experiments will be added incrementally.

---

## Architecture

```text
BERKELIUM
│
├── 01-SYSTEM-AMD-GPU
├── 02-PYTHON-ENVIRONMENT
├── 03-AMD-DEEP-LEARNING
├── 04-LLM-TRAINING
├── 05-LONG-CONTEXT
├── 06-CODING-DATASET-ENGINEERING
├── 07-CODE-INTELLIGENCE
├── 08-OCR-VISION
├── 09-3D-NEURAL-RENDERING
├── 10-EXR-HDR
├── 11-VIDEO-TEMPORAL-DATA
├── 12-DATA-FORMATS-TRAINING-SHARDS
├── 13-TRAINING-EXPERIMENTATION
├── 14-DISTRIBUTED-TRAINING
└── 15-DEVELOPMENT-RESEARCH

Bro, since the repo is brand new, don’t write a 10-page README yet. 😂

I’d make the README establish what Berkelium is, what it is intended to become, and the architecture — without pretending everything is already implemented.

Paste this as the initial README.md:

# BERKELIUM — Neural Rendering & Sciences
> An experimental research and engineering repository for AI, neural rendering, 3D vision, scientific computing, and AMD GPU-accelerated workloads.
**Status:** 🚧 Early-stage research / infrastructure skeleton
---

---
## Architecture
```text
BERKELIUM
│
├── 01-SYSTEM-AMD-GPU
├── 02-PYTHON-ENVIRONMENT
├── 03-AMD-DEEP-LEARNING
├── 04-LLM-TRAINING
├── 05-LONG-CONTEXT
├── 06-CODING-DATASET-ENGINEERING
├── 07-CODE-INTELLIGENCE
├── 08-OCR-VISION
├── 09-3D-NEURAL-RENDERING
├── 10-EXR-HDR
├── 11-VIDEO-TEMPORAL-DATA
├── 12-DATA-FORMATS-TRAINING-SHARDS
├── 13-TRAINING-EXPERIMENTATION
├── 14-DISTRIBUTED-TRAINING
└── 15-DEVELOPMENT-RESEARCH

The architecture is intentionally modular. Not every component is required for every workload.

⸻

Data Architecture

Berkelium is designed to support multimodal and large-scale training data pipelines.

RAW DATA
   │
   ├── JSONL
   │     ├── Metadata
   │     ├── Tasks
   │     └── Annotations
   │
   ├── PNG / JPEG
   │     └── RGB Images
   │
   ├── EXR
   │     ├── Depth
   │     ├── Surface Normals
   │     └── HDR Rendering Targets
   │
   ├── MP4 / WebM
   │     └── Temporal Rendering Sequences
   │
   └── 3D / Scientific Data
          │
          ▼
     Validation / Cleaning
          │
          ▼
   Parquet / WebDataset
          │
          ▼
    Training / Evaluation

⸻

AMD GPU & ROCm

Berkelium is designed with AMD GPU acceleration in mind, including:

* ROCm
* HIP
* PyTorch on ROCm
* RCCL
* GPU profiling
* Distributed training
* GPU benchmarking and optimization

Primary development and experimentation may target AMD Instinct accelerators where available.

⸻

Research Areas

Neural Rendering

Research into:

* Neural scene representations
* 3D reconstruction
* Differentiable rendering
* Synthetic data generation
* Depth and normal estimation
* Multiview reconstruction
* Temporal rendering

Computer Vision & OCR

Research into:

* Image understanding
* Object detection
* Segmentation
* OCR
* Vision-language models
* Synthetic training data

3D & Geometry

Research into:

* Mesh processing
* 3D reconstruction
* Geometry learning
* Neural geometry
* Retopology
* Blender-based synthetic data generation

Scientific Computing

Potential applications include:

* Space science
* Planetary visualization
* Earth observation
* Scientific simulation
* Scientific data analysis

⸻

Experiments

Experiments will be kept reproducible wherever possible.

Each experiment should document:

1. Dataset and data source
2. Preprocessing
3. Model / algorithm
4. Hardware
5. Software environment
6. Training configuration
7. Evaluation methodology
8. Results
9. Limitations
10. Reproducibility information

Original measurements, benchmarks, experiments, and derived analyses will be documented separately from external datasets, models, and software.

⸻

Data & Attribution

External datasets, models, software, papers, APIs, and other resources will be documented with their respective sources and licenses.

Berkelium does not claim ownership of external resources incorporated into experiments.

Original experiments, measurements, benchmark results, and derived analyses produced within the project will be documented as project research outputs, with the underlying methodology and source materials recorded where applicable.

⸻

Development Philosophy

Berkelium is built incrementally.

The repository may contain experimental, incomplete, or research-stage components. A component should not be considered production-ready unless its implementation, testing, documentation, and evaluation support that claim.

The goal is not to collect every possible AI tool.

The goal is to build a coherent research and engineering stack that can support increasingly demanding AI, 3D, rendering, and scientific workloads.

⸻

Roadmap

Phase 1 — Foundation

* [x]	Repository architecture
* [ ]	Development environment
* [ ]	AMD / ROCm environment validation
* [ ]	Dependency management
* [ ]	Basic benchmarking

Phase 2 — Data Infrastructure

* [ ]	Dataset schemas
* [ ]	Data validation
* [ ]	JSONL pipelines
* [ ]	Image pipelines
* [ ]	EXR/HDR pipelines
* [ ]	Video pipelines
* [ ]	Parquet/WebDataset sharding

Phase 3 — AI & Vision

* [ ]	OCR pipeline
* [ ]	Vision models
* [ ]	Multimodal experiments
* [ ]	Evaluation framework

Phase 4 — 3D & Neural Rendering

* [ ]	3D data pipeline
* [ ]	Rendering pipeline
* [ ]	Neural rendering experiments
* [ ]	Geometry learning
* [ ]	AI-assisted retopology research

Phase 5 — Distributed & Large-Scale Training

* [ ]	Multi-GPU training
* [ ]	Distributed data loading
* [ ]	Model optimization
* [ ]	Large-scale benchmarking

⸻

Current Status

This repository is currently an architectural foundation.

Implementation will be added progressively as individual research and engineering objectives are defined.

⸻

License

License information will be added as the project matures.


