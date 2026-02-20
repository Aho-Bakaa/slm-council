# ─────────────────────────────────────────────────────────────────
# SLM Coding Council – GCP Infrastructure (Terraform)
# ─────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    # Configure in terraform init:
    #   terraform init -backend-config="bucket=YOUR_BUCKET" -backend-config="prefix=slm-council"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─────────────────────────────────────────────────────────────────
# Enable required APIs
# ─────────────────────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# ─────────────────────────────────────────────────────────────────
# Artifact Registry – Docker images
# ─────────────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "slm_council" {
  repository_id = "slm-council"
  location      = var.region
  format        = "DOCKER"
  description   = "Docker images for SLM Coding Council"

  depends_on = [google_project_service.apis]
}

# ─────────────────────────────────────────────────────────────────
# Service Account for the council workloads
# ─────────────────────────────────────────────────────────────────

resource "google_service_account" "council_sa" {
  account_id   = "slm-council-sa"
  display_name = "SLM Coding Council Service Account"
}

resource "google_project_iam_member" "council_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.council_sa.email}"
}

# ─────────────────────────────────────────────────────────────────
# Cloud Run – API Gateway
# ─────────────────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "gateway" {
  name     = "slm-council-gateway"
  location = var.region

  template {
    service_account = google_service_account.council_sa.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/slm-council/gateway:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      dynamic "env" {
        for_each = var.gateway_env_vars
        content {
          name  = env.key
          value = env.value
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }
  }

  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access (or restrict as needed)
resource "google_cloud_run_v2_service_iam_member" "gateway_public" {
  count    = var.allow_public_access ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.gateway.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ─────────────────────────────────────────────────────────────────
# VPC Network (for private agent endpoints)
# ─────────────────────────────────────────────────────────────────

resource "google_compute_network" "council_vpc" {
  name                    = "slm-council-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "council_subnet" {
  name          = "slm-council-subnet"
  region        = var.region
  network       = google_compute_network.council_vpc.id
  ip_cidr_range = "10.0.0.0/24"
}
