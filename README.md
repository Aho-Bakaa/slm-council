# SLM Coding Council

> **Many-to-One** autonomous coding & research engine – a high-precision,
> cost-effective pipeline where a central reasoning **Brain** orchestrates four
> specialised Small Language Models to handle complex software engineering tasks.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │        User Request          │
                    └─────────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │    Orchestrator (Qwen3-Max)  │
                    │    "The One" – Decomposes    │
                    └──┬──────┬──────┬──────┬─────┘
                       │      │      │      │
              ┌────────┘      │      │      └────────┐
              ▼               ▼      ▼               ▼
     ┌────────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
     │  Researcher    │ │ Generator│ │ Debugger │ │   Tester     │
     │  Gemma 3 4B-IT │ │ Qwen3-   │ │ DeepSeek │ │  Phi-4-mini  │
     │                │ │ Coder 4B │ │ R1 7B    │ │              │
     └───────┬────────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
             │               │            │               │
             └───────────────┴─────┬──────┴───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  Orchestrator: SYNTHESISE    │
                    │  APPROVE  ←→  REFINE (loop) │
                    └─────────────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │     Final Council Result     │
                    └─────────────────────────────┘
```

### Pipeline Flow

1. **Decompose** – The Orchestrator breaks the user's request into agent tasks
2. **Research** – Tech Researcher scouts docs, APIs, and best practices → outputs a **Tech Manifest**
3. **Generate** – Code Generator consumes the manifest and writes production code
4. **Review** – Debugger (CoT reasoning) and Tester (unit tests) run **in parallel**
5. **Synthesise** – Orchestrator evaluates all reports:
   - **APPROVE** → ship the code
   - **REFINE** → targeted feedback sent back to specific agents (up to N passes)

---

## Components

| Role | Model | GPU | Purpose |
|------|-------|-----|---------|
| **Orchestrator** | Qwen3-Max / 72B | Vertex AI | Project Manager – decomposes, synthesises, decides |
| **Tech Researcher** | Gemma 3 4B-IT | L4 | Documentation & architecture scout |
| **Code Generator** | Qwen3-Coder 4B | L4 | Writes clean, type-hinted code |
| **Debugger** | DeepSeek-R1-Distill-Qwen-7B | L4 | CoT bug hunter & logic tracer |
| **Tester** | Phi-4-mini | L4 | Adversarial QA – generates tests & edge cases |

---

## Project Structure

```
slm_council/
├── src/slm_council/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── config.py            # Pydantic settings from .env
│   ├── models.py            # All domain models & schemas
│   ├── server.py            # FastAPI application
│   ├── agents/
│   │   ├── base.py          # Abstract agent (HTTP + retry + JSON parsing)
│   │   ├── researcher.py    # Tech Researcher (Gemma 3 4B)
│   │   ├── generator.py     # Code Generator (Qwen3-Coder 4B)
│   │   ├── debugger.py      # Debugger (DeepSeek-R1-Distill)
│   │   └── tester.py        # Tester (Phi-4-mini)
│   ├── orchestrator/
│   │   ├── brain.py         # Central reasoning engine
│   │   └── loop.py          # Iterative refinement loop
│   └── utils/
│       ├── logging.py       # Structured logging (structlog)
│       └── prompts.py       # All system & task prompts
├── tests/
│   ├── test_models.py
│   ├── test_agents.py
│   └── test_orchestrator.py
├── deploy/
│   ├── vertex_ai/
│   │   ├── deploy_agents.py # Script to deploy SLMs to Vertex AI
│   │   └── endpoint_config.yaml
│   └── terraform/
│       ├── main.tf          # GCP infra (Cloud Run, VPC, IAM, etc.)
│       ├── variables.tf
│       └── outputs.tf
├── Dockerfile               # API gateway container
├── Dockerfile.vllm          # vLLM model server template
├── docker-compose.yml       # Local dev with all 4 agents + gateway
├── cloudbuild.yaml          # CI/CD pipeline
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url> && cd slm_council
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your GCP project ID and endpoint URLs
```

### 3. Run locally (with Docker Compose)

> Requires NVIDIA Container Toolkit for GPU-backed agents.

```bash
docker compose up --build
```

The gateway will be available at `http://localhost:8080`.

### 4. Run the API server only (agents hosted elsewhere)

```bash
python -m slm_council
# or
uvicorn slm_council.server:app --host 0.0.0.0 --port 8080
```

### 5. Test the pipeline

```bash
curl -X POST http://localhost:8080/council/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Build a Python function that implements binary search on a sorted list",
    "language": "python"
  }'
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/council/run` | Submit a task to the full council pipeline |
| `GET` | `/config/agents` | View current agent configuration |

### POST `/council/run`

**Request:**
```json
{
  "query": "Build a REST API for user management with CRUD operations",
  "language": "python",
  "context": {}
}
```

**Response:**
```json
{
  "session_id": "a1b2c3d4e5f6",
  "status": "pass",
  "code": {
    "files": [...],
    "explanation": "...",
    "assumptions": [...]
  },
  "debug_report": { "verdict": "pass", "bugs": [] },
  "test_report": { "verdict": "pass", "test_cases": [...] },
  "tech_manifest": { "summary": "...", "dependencies": [...] },
  "refinement_passes": 1,
  "total_duration_secs": 45.2,
  "summary": "All checks passed. Code is production-ready."
}
```

---

## Deploy to GCP

### Option A: Terraform (recommended)

```bash
cd deploy/terraform
terraform init -backend-config="bucket=YOUR_TF_STATE_BUCKET"
terraform plan -var="project_id=YOUR_PROJECT"
terraform apply -var="project_id=YOUR_PROJECT"
```

### Option B: Manual deployment

```bash
# 1. Deploy SLM agents to Vertex AI
python deploy/vertex_ai/deploy_agents.py --project YOUR_PROJECT --region us-central1

# 2. Build & deploy the gateway via Cloud Build
gcloud builds submit --config cloudbuild.yaml --project YOUR_PROJECT
```

---

## Configuration

All settings are loaded from environment variables (or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | - | GCP project ID |
| `GCP_REGION` | `us-central1` | GCP region |
| `ORCHESTRATOR_MODEL_NAME` | `qwen3-max` | Orchestrator model identifier |
| `RESEARCHER_ENDPOINT` | `http://localhost:8001/v1` | Researcher vLLM endpoint |
| `GENERATOR_ENDPOINT` | `http://localhost:8002/v1` | Generator vLLM endpoint |
| `DEBUGGER_ENDPOINT` | `http://localhost:8003/v1` | Debugger vLLM endpoint |
| `TESTER_ENDPOINT` | `http://localhost:8004/v1` | Tester vLLM endpoint |
| `MAX_REFINEMENT_PASSES` | `3` | Max REFINE iterations before returning best-effort |
| `CONSENSUS_THRESHOLD` | `0.8` | Quality threshold for auto-approval |
| `REQUEST_TIMEOUT_SECS` | `120` | Per-agent request timeout |

---

## Running Tests

```bash
pytest tests/ -v --cov=slm_council
```

---

## Cost Optimisation

The Many-to-One architecture is designed for cost efficiency:

- **4B parameter SLMs** on single L4 GPUs (~$0.70/hr each on GCP)
- Only the Orchestrator uses a large model (and it doesn't write bulk code)
- Agents scale to zero when idle (via Vertex AI auto-scaling)
- Refinement passes are capped to prevent runaway costs

**Estimated cost per query** (assuming ~2 min total pipeline time):
- 4 × L4 agents: ~$0.09
- 1 × Orchestrator (API): ~$0.02
- **Total: ~$0.11/query**

---

## License

MIT
