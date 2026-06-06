# ---------------------------------------------------------------------------
# S3 artifacts bucket
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "pour-decisions-artifacts-"
  force_destroy = true

  tags = {
    Name        = "pour-decisions-artifacts"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

# SSE-S3 (AES256) — free tier encryption, no KMS cost.
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# Archive: zip the repo and ship it to S3 so instances can pull it.
# data/prepared/ is intentionally INCLUDED (training data lives there).
# ---------------------------------------------------------------------------
data "archive_file" "repo" {
  type        = "zip"
  source_dir  = "${path.module}/.."
  output_path = "${path.module}/repo.zip"

  excludes = [
    "**/.venv",
    "**/.git",
    "**/node_modules",
    "**/models",
    "**/merged",
    "serving/merged",
    "**/.terraform",
    "**/mlruns",
    "**/__pycache__",
    "**/reports",
    "infra/repo.zip",
    "infra/.terraform",
    "**/*.pem",
    "CONTEXT.local.md",
    "kaggle.json",
    "**/.env",
  ]
}

resource "aws_s3_object" "repo_zip" {
  bucket = aws_s3_bucket.artifacts.id
  key    = "repo.zip"
  source = data.archive_file.repo.output_path
  etag   = data.archive_file.repo.output_md5

  tags = {
    Name      = "pour-decisions-repo-zip"
    ManagedBy = "opentofu"
    Project   = "pour-decisions"
  }
}

# ---------------------------------------------------------------------------
# Networking — minimal public VPC (no default VPC exists in this account)
# SSM is outbound-only HTTPS; public subnet gives instances egress via IGW.
# No ingress rules; the public IP is for egress, not inbound access.
# ---------------------------------------------------------------------------
data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name        = "pour-decisions"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[var.az_index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "pour-decisions-public"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "pour-decisions"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "pour-decisions-public"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# ---------------------------------------------------------------------------
# Security group: no ingress, full egress
# SSM uses outbound HTTPS to regional endpoints; no inbound port needed.
# ---------------------------------------------------------------------------
resource "aws_security_group" "instances" {
  name_prefix = "pour-decisions-instances-"
  description = "SSM-managed instances: self-ingress 8000 (eval), egress all"
  vpc_id      = aws_vpc.main.id

  # Self-ingress on 8000: allows the tuned serve box to reach the base serve
  # box's vLLM endpoint for eval comparison. No public ingress; SSM is outbound HTTPS.
  ingress {
    description = "vLLM port: serve instances can reach each other for eval"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    self        = true
  }

  egress {
    description      = "Allow all outbound traffic"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name        = "pour-decisions-instances"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# IAM: instance role + profile for SSM + scoped S3 access
# name_prefix avoids collisions in a shared corporate account (IAM is global).
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "instance_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "artifacts_s3" {
  statement {
    sid    = "BucketList"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.artifacts.arn,
    ]
  }

  statement {
    sid    = "ObjectAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = [
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }
}

resource "aws_iam_role" "instance" {
  name_prefix        = "pour-decisions-instance-"
  assume_role_policy = data.aws_iam_policy_document.instance_assume_role.json

  tags = {
    Name        = "pour-decisions-instance"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "artifacts_s3" {
  name_prefix = "pour-decisions-artifacts-s3-"
  role        = aws_iam_role.instance.id
  policy      = data.aws_iam_policy_document.artifacts_s3.json
}

resource "aws_iam_instance_profile" "instance" {
  name_prefix = "pour-decisions-instance-"
  role        = aws_iam_role.instance.name

  tags = {
    Name      = "pour-decisions-instance"
    ManagedBy = "opentofu"
    Project   = "pour-decisions"
  }
}

# ---------------------------------------------------------------------------
# AMI: Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)
# Pinned to PyTorch 2.7 to prevent silent version drift on re-apply.
# Review this pin when upgrading PyTorch (run: aws ec2 describe-images --owners amazon
#   --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.*...")
# ---------------------------------------------------------------------------
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)*"]
  }
}

# ---------------------------------------------------------------------------
# Train instance (g6e.xlarge — L40S 48GB)
# Spot with persistent/stop: on reclaim AWS stops the instance and preserves the
# root EBS (model data, LoRA checkpoints). The spot request auto-re-queues so the
# instance restarts when capacity is available. The systemd pour-train.service unit
# (installed by bootstrap) fires on every boot; the .train-done guard stops it
# re-running after a successful job.
#
# root_block_device.delete_on_termination = false: the provider default is true,
# but spot-stop preserves volumes regardless. Setting false here makes the intent
# explicit and guards against accidental termination blowing away the EBS.
# ---------------------------------------------------------------------------
resource "aws_instance" "train" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.train_instance
  iam_instance_profile   = aws_iam_instance_profile.instance.name
  vpc_security_group_ids = [aws_security_group.instances.id]
  subnet_id              = aws_subnet.public.id

  user_data = templatefile("${path.module}/templates/train-bootstrap.sh.tftpl", {
    bucket_name = aws_s3_bucket.artifacts.id
    region      = var.region
  })
  user_data_replace_on_change = true # bootstrap runs once per instance; edits must replace, not stop/start

  root_block_device {
    volume_size           = 200
    encrypted             = true
    delete_on_termination = false
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        # persistent + stop: EBS survives reclaim; request re-queues automatically.
        # Requires interruption_behavior = "stop" (not "terminate").
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  tags = {
    Name        = "pour-decisions-train"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

# ---------------------------------------------------------------------------
# Serve instances (g5.2xlarge — A10G 24GB, 32GB RAM)
# Both gated behind var.serve_enabled (default false). Two separate boxes so
# each 7B model gets its own dedicated A10G GPU — no contention.
#
# serve (tuned): serves the LoRA-merged model; also runs make eval.
# serve_base:    serves the base model; tuned box calls :8000 over private net.
# ---------------------------------------------------------------------------
resource "aws_instance" "serve" {
  count = var.serve_enabled ? 1 : 0

  ami                    = data.aws_ami.dlami.id
  instance_type          = var.serve_instance
  iam_instance_profile   = aws_iam_instance_profile.instance.name
  vpc_security_group_ids = [aws_security_group.instances.id]
  subnet_id              = aws_subnet.public.id

  user_data = templatefile("${path.module}/templates/serve-bootstrap.sh.tftpl", {
    bucket_name       = aws_s3_bucket.artifacts.id
    region            = var.region
    model_s3_prefix   = "merged"
    served_model_name = "cocktail-tuned"
    is_eval_host      = true
  })
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 100
    encrypted   = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
    }
  }

  tags = {
    Name        = "pour-decisions-serve-tuned"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}

resource "aws_instance" "serve_base" {
  count = var.serve_enabled ? 1 : 0

  ami                    = data.aws_ami.dlami.id
  instance_type          = var.serve_instance
  iam_instance_profile   = aws_iam_instance_profile.instance.name
  vpc_security_group_ids = [aws_security_group.instances.id]
  subnet_id              = aws_subnet.public.id

  user_data = templatefile("${path.module}/templates/serve-bootstrap.sh.tftpl", {
    bucket_name       = aws_s3_bucket.artifacts.id
    region            = var.region
    model_s3_prefix   = "models/7B-Instruct-v0.3"
    served_model_name = "cocktail-base"
    is_eval_host      = false
  })
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 100
    encrypted   = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
    }
  }

  tags = {
    Name        = "pour-decisions-serve-base"
    Environment = "ml"
    ManagedBy   = "opentofu"
    Project     = "pour-decisions"
  }
}
