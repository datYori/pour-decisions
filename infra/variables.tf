variable "region" {
  type        = string
  description = "AWS region for all resources."
  default     = "eu-central-1" # Frankfurt: only EU region with g6e + g5 + nonzero G quota (spot G quota is 0 EU-wide)
}

variable "train_instance" {
  type        = string
  description = "EC2 instance type for the training node."
  default     = "g5.2xlarge" # 1x A10G 24GB + 32GB RAM. g5.xlarge (16GB RAM) thrashes loading the 14.5GB model + bf16 copy on CPU; 32GB clears it. (g6e.xlarge L40S has no on-demand capacity here.)
}

variable "serve_instance" {
  type        = string
  description = "EC2 instance type for the serving node."
  default     = "g5.2xlarge" # 1x A10G 24GB + 32GB RAM. g5.xlarge (16GB RAM) thrashes loading the 14.5GB 7B model; 32GB clears it.
}

variable "use_spot" {
  type        = bool
  description = "Whether to launch instances as Spot. Train box uses persistent/stop so EBS survives reclaim and the request auto-re-queues. Serve box uses one-time/terminate (stateless). Default false: Spot G/VT quota is 0 across EU regions; on-demand G quota (768) is available in Frankfurt."
  default     = false
}

variable "az_index" {
  type        = number
  description = "Index into the region's AZ list for the subnet. GPU on-demand capacity is AZ-specific; eu-central-1a (index 0) had no g5 capacity, 1b (index 1) did. Bump if RunInstances returns InsufficientInstanceCapacity."
  default     = 1
}

variable "serve_enabled" {
  type        = bool
  description = "Whether to create the serve instance. Default false — the serve box is useless until merged artifacts are uploaded to S3 after training."
  default     = false
}
