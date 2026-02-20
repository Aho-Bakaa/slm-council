"""Deploy SLM agents to Vertex AI Prediction Endpoints.

This script:
1. Uploads each vLLM model container to Vertex AI Model Registry.
2. Creates GPU-backed endpoints for each agent.
3. Deploys the models to their endpoints.

Prerequisites:
  - `gcloud auth application-default login`
  - GCP project with Vertex AI API enabled
  - L4 GPU quota in the target region

Usage:
  python deploy/vertex_ai/deploy_agents.py --project YOUR_PROJECT --region us-central1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from google.cloud import aiplatform


@dataclass
class AgentSpec:
    name: str
    model_id: str
    display_name: str
    port: int = 8000
    machine_type: str = "g2-standard-8"       # 1× L4 GPU
    accelerator_type: str = "NVIDIA_L4"
    accelerator_count: int = 1
    min_replicas: int = 1
    max_replicas: int = 2
    container_image: str = "vllm/vllm-openai:latest"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90


# ── Agent definitions ────────────────────────────────────────────

AGENTS: list[AgentSpec] = [
    AgentSpec(
        name="researcher",
        model_id="google/gemma-3-4b-it",
        display_name="SLM Council – Tech Researcher (Gemma 3 4B-IT)",
    ),
    AgentSpec(
        name="generator",
        model_id="Qwen/Qwen3-Coder-4B",
        display_name="SLM Council – Code Generator (Qwen3-Coder 4B)",
    ),
    AgentSpec(
        name="debugger",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        display_name="SLM Council – Debugger (DeepSeek-R1-Distill-Qwen-7B)",
        machine_type="g2-standard-12",  # 7B needs a bit more RAM
    ),
    AgentSpec(
        name="tester",
        model_id="microsoft/phi-4-mini-instruct",
        display_name="SLM Council – Tester (Phi-4-mini)",
    ),
]


def deploy_agent(spec: AgentSpec, project: str, region: str) -> None:
    """Upload model + create endpoint + deploy for a single agent."""
    print(f"\n{'='*60}")
    print(f"Deploying: {spec.display_name}")
    print(f"{'='*60}")

    # 1. Upload model to registry
    env_vars = {
        "MODEL_ID": spec.model_id,
        "MAX_MODEL_LEN": str(spec.max_model_len),
        "GPU_MEMORY_UTILIZATION": str(spec.gpu_memory_utilization),
    }

    model = aiplatform.Model.upload(
        display_name=f"slm-council-{spec.name}",
        serving_container_image_uri=spec.container_image,
        serving_container_ports=[spec.port],
        serving_container_environment_variables=env_vars,
        serving_container_command=[
            "python", "-m", "vllm.entrypoints.openai.api_server",
        ],
        serving_container_args=[
            "--model", spec.model_id,
            "--max-model-len", str(spec.max_model_len),
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            "--host", "0.0.0.0",
            "--port", str(spec.port),
        ],
    )
    print(f"  Model uploaded: {model.resource_name}")

    # 2. Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=f"slm-council-{spec.name}-endpoint",
    )
    print(f"  Endpoint created: {endpoint.resource_name}")

    # 3. Deploy model to endpoint
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"slm-council-{spec.name}-deployed",
        machine_type=spec.machine_type,
        accelerator_type=spec.accelerator_type,
        accelerator_count=spec.accelerator_count,
        min_replica_count=spec.min_replicas,
        max_replica_count=spec.max_replicas,
        traffic_percentage=100,
    )
    print(f"  Deployed to: {endpoint.resource_name}")
    print(f"  Endpoint URL: https://{region}-aiplatform.googleapis.com/v1/{endpoint.resource_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy SLM agents to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument(
        "--agents",
        nargs="*",
        choices=["researcher", "generator", "debugger", "tester"],
        default=["researcher", "generator", "debugger", "tester"],
        help="Which agents to deploy (default: all)",
    )
    args = parser.parse_args()

    aiplatform.init(project=args.project, location=args.region)

    agent_map = {a.name: a for a in AGENTS}
    for name in args.agents:
        deploy_agent(agent_map[name], args.project, args.region)

    print("\n✓ All agents deployed successfully.")
    print("  Update your .env with the endpoint URLs above.")


if __name__ == "__main__":
    main()
