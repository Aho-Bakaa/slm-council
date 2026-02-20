variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP region for all resources"
}

variable "allow_public_access" {
  type        = bool
  default     = false
  description = "Whether to allow unauthenticated access to the gateway"
}

variable "gateway_env_vars" {
  type        = map(string)
  default     = {}
  description = "Environment variables to inject into the Cloud Run gateway"
}
