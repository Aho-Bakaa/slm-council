# SLM Coding Council

Many-to-One autonomous coding engine with a dynamic DAG orchestrator, specialist agents, and iterative refinement.

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

## Current configuration status

Code supports 8 roles, but default settings currently preconfigure the original 4 endpoint-backed specialists:

- Researcher
- Generator
- Debugger
- Tester

Additional roles (planner/reviewer/optimizer/refactorer) are wired in orchestration and schemas, and can be activated by adding their settings fields/values in `src/slm_council/config.py` and `.env`.

## Core flow

1. Request enters `POST /council/run`
2. Orchestrator creates DAG tasks (`id`, `agent`, `instruction`, `dependencies`)
3. DAG executor validates graph (duplicate IDs, missing deps, cycles)
4. Tasks execute in topological waves with semaphore-based concurrency limits
5. Outputs are propagated to shared context for dependent tasks
6. Orchestrator synthesizes all reports and decides APPROVE or REFINE
7. On REFINE, orchestrator emits refinement DAG and loop continues (bounded by max passes)
8. Final `CouncilResult` includes outputs + execution metadata

## Project structure

```text
slm_council/
├── src/slm_council/
│   ├── config.py
│   ├── models.py
│   ├── server.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── researcher.py
│   │   ├── planner.py
│   │   ├── generator.py
│   │   ├── reviewer.py
│   │   ├── debugger.py
│   │   ├── tester.py
│   │   ├── optimizer.py
│   │   └── refactorer.py
│   ├── orchestrator/
│   │   ├── brain.py
│   │   ├── dag.py
│   │   └── loop.py
│   ├── knowledge/
│   │   ├── base.py
│   │   └── style.py
│   ├── feedback/
│   │   └── base.py
│   └── utils/
│       ├── prompts.py
│       └── logging.py
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | basic service info |
| GET | `/health` | health check |
| POST | `/council/run` | run full council loop |
| GET | `/config/agents` | runtime model/endpoint config visibility |

