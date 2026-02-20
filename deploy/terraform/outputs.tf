output "gateway_url" {
  value       = google_cloud_run_v2_service.gateway.uri
  description = "URL of the SLM Council API gateway"
}

output "service_account_email" {
  value       = google_service_account.council_sa.email
  description = "Service account used by the council workloads"
}

output "artifact_registry" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.slm_council.repository_id}"
  description = "Docker registry path"
}

output "vpc_network" {
  value       = google_compute_network.council_vpc.name
  description = "VPC network for agent endpoints"
}
