output "artifacts_bucket" {
  description = "Name of the S3 artifacts bucket."
  value       = aws_s3_bucket.artifacts.bucket
}

output "train_instance_id" {
  description = "EC2 instance ID of the training node."
  value       = aws_instance.train.id
}

output "serve_tuned_instance_id" {
  description = "EC2 instance ID of the tuned serve node (merged model). Null when var.serve_enabled=false."
  value       = var.serve_enabled ? aws_instance.serve[0].id : null
}

output "serve_base_instance_id" {
  description = "EC2 instance ID of the base serve node. Null when var.serve_enabled=false."
  value       = var.serve_enabled ? aws_instance.serve_base[0].id : null
}

output "serve_base_private_ip" {
  description = "Private IP of the base serve node (used by eval on the tuned box). Null when var.serve_enabled=false."
  value       = var.serve_enabled ? aws_instance.serve_base[0].private_ip : null
}

output "train_ssm_command" {
  description = "AWS CLI command to trigger fine-tuning via SSM. Creates the .train-requested marker then starts pour-train.service. On a subsequent spot stop+restart, systemd fires the service automatically (the .train-done guard prevents re-runs after success)."
  value       = "aws ssm send-command --instance-ids ${aws_instance.train.id} --document-name \"AWS-RunShellScript\" --parameters 'commands=[\"touch /opt/pour-decisions/.train-requested && systemctl start pour-train.service\"]' --region ${var.region} --profile default"
}

output "serve_tuned_ssm_command" {
  description = "AWS CLI command to start vLLM on the tuned serve box (merged model, cocktail-tuned). Empty when var.serve_enabled=false."
  value       = var.serve_enabled ? "aws ssm send-command --instance-ids ${aws_instance.serve[0].id} --document-name \"AWS-RunShellScript\" --parameters 'commands=[\"bash /opt/pour-decisions/serve.sh\"]' --region ${var.region} --profile default" : ""
}

output "serve_base_ssm_command" {
  description = "AWS CLI command to start vLLM on the base serve box (base model, cocktail-base). Empty when var.serve_enabled=false."
  value       = var.serve_enabled ? "aws ssm send-command --instance-ids ${aws_instance.serve_base[0].id} --document-name \"AWS-RunShellScript\" --parameters 'commands=[\"bash /opt/pour-decisions/serve.sh\"]' --region ${var.region} --profile default" : ""
}

output "eval_ssm_command" {
  description = "AWS CLI command to run make eval on the tuned box. Targets base model at serve_base_private_ip:8000, tuned model at localhost:8000. Empty when var.serve_enabled=false."
  value       = var.serve_enabled ? "aws ssm send-command --instance-ids ${aws_instance.serve[0].id} --document-name \"AWS-RunShellScript\" --parameters 'commands=[\"cd /opt/pour-decisions && BASE_MODEL=cocktail-base TUNED_MODEL=cocktail-tuned BASE_URL=http://${aws_instance.serve_base[0].private_ip}:8000/v1 TUNED_URL=http://localhost:8000/v1 make eval\"]' --region ${var.region} --profile default" : ""
}
