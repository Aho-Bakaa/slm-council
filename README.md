# SLM Coding Council

Many-to-One autonomous coding engine with a dynamic DAG orchestrator, specialist agents, and iterative refinement.

## What changed

The architecture has moved from a fixed 4-agent sequence to a dynamic graph executor:

- Orchestrator now decomposes work into DAG tasks with dependencies
- DAG executor runs dependency-safe waves in parallel
- Agent registry enables plug-in style agent discovery
- System supports 8 specialist roles at the model/schema/orchestration level
- Final response includes richer artifacts (architecture/review/optimization/refactor reports)

## Agent roles

At runtime, roles are discovered through the registry and instantiated from settings.

| Role | Primary Output | Purpose |
|---|---|---|
| orchestrator | task DAG + verdict | Planning, synthesis, refine/approve decisions |
| researcher | TechManifest | dependencies, constraints, references |
| planner | ArchitecturePlan | components, file layout, data flow |
| generator | GeneratedCode | implementation files |
| reviewer | ReviewReport | style/security/best-practice review |
| debugger | DebugReport | bug tracing and fixes |
| tester | TestReport | tests, edge cases, coverage notes |
| optimizer | OptimizationReport | complexity and bottleneck analysis |
| refactorer | RefactorResult | structural cleanup while preserving behavior |



## Core flow

1. Request enters `POST /council/run`
2. Orchestrator creates DAG tasks (`id`, `agent`, `instruction`, `dependencies`)
3. DAG executor validates graph (duplicate IDs, missing deps, cycles)
4. Tasks execute in topological waves with semaphore-based concurrency limits
5. Outputs are propagated to shared context for dependent tasks
6. Orchestrator synthesizes all reports and decides APPROVE or REFINE
7. On REFINE, orchestrator emits refinement DAG and loop continues (bounded by max passes)
8. Final `CouncilResult` includes outputs + execution metadata

## Quick start

### 1) Install

```bash
git clone <repo-url>
cd slm_council
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

### 2) Configure

```bash
cp .env.example .env
```

Update `.env` with orchestrator and agent endpoints/models.

### 3) Run

```bash
python -m slm_council
# or
```




