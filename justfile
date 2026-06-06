region              := "eu-central-1"
profile             := "default"
infra_dir           := "infra"
mistral_finetune_dir := env_var_or_default("MISTRAL_FINETUNE_DIR", env_var("HOME") + "/mistral-finetune")
base_model_dir       := env_var_or_default("BASE_MODEL_DIR",       env_var("HOME") + "/m7b-base")
adapter_dir          := env_var_or_default("ADAPTER_DIR",          env_var("HOME") + "/cocktail-adapter")
merged_dir           := env_var_or_default("MERGED_DIR",           env_var("HOME") + "/merged")

# List all recipes
default:
    @just --list

# ── dev ───────────────────────────────────────────────────────────────────────

install:
    uv sync --extra dev

test:
    uv run pytest -q

lint:
    uv run ruff check . && uv run mypy src

data:
    uv run python -m pour_decisions.build_data

# ── local matrix (Apple Silicon / MLX; CUDA via backend=peft on a workstation) ──────────────

# Install local fine-tune deps (keeps dev tools). backend=mlx (Mac) or peft (CUDA workstation).
local-setup backend="mlx":
    uv sync --extra {{backend}} --extra dev
    @echo "Gated models (gemma-*, llama-3.2-1b) need: hf auth login + accept the license on HF."

# Fine-tune + eval one model (dev loop). Positional args: just local-one qwen2.5-0.5b
local-one key backend="mlx" modes="unconstrained,constrained":
    uv run python -m pour_decisions.matrix.run_matrix --backend {{backend}} --key {{key}} --modes {{modes}}

# Run the whole matrix.
local-matrix backend="mlx" modes="unconstrained,constrained":
    uv run python -m pour_decisions.matrix.run_matrix --backend {{backend}} --modes {{modes}}

# Re-render the report from the committed json (no retrain).
local-report:
    uv run python -c "from pathlib import Path; from pour_decisions.matrix.report import load_results, rebuild_results, render_markdown; print(render_markdown(rebuild_results(load_results(Path('reports/local-matrix.json')))))"

# ── infra ─────────────────────────────────────────────────────────────────────

# Apply infra and wait for the train instance to finish bootstrapping
infra:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{infra_dir}} && tofu init && AWS_PROFILE={{profile}} tofu apply
    just _wait-ready "$(cd {{infra_dir}} && tofu output -raw train_instance_id)" "train"

# Enable serve instances and wait for both to finish bootstrapping
infra-serve:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{infra_dir}} && AWS_PROFILE={{profile}} tofu apply -var serve_enabled=true
    just _wait-ready "$(cd {{infra_dir}} && tofu output -raw serve_tuned_instance_id)" "serve-tuned"
    just _wait-ready "$(cd {{infra_dir}} && tofu output -raw serve_base_instance_id)" "serve-base"

# Install mistral-finetune with pinned deps (required for local merge-lora workflow)
# Override location: MISTRAL_FINETUNE_DIR=~/path just setup-mistral-finetune
setup-mistral-finetune:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR="{{mistral_finetune_dir}}"
    if [[ ! -d "$DIR" ]]; then
        git clone https://github.com/mistralai/mistral-finetune.git "$DIR"
    else
        echo "Already cloned at $DIR — skipping clone."
    fi
    cd "$DIR"
    uv venv --python 3.11
    uv pip install torch==2.2 triton==2.2 "numpy<2"
    uv pip install -r requirements.txt --no-build-isolation
    uv pip install "mistral-common==1.4.4"
    echo "mistral-finetune ready at $DIR"

# Merge LoRA adapter into base weights locally (~32GB RAM, no GPU needed)
# Requires: just setup-mistral-finetune + hf download of base + adapter
# Override paths via env: BASE_MODEL_DIR, ADAPTER_DIR, MERGED_DIR, MISTRAL_FINETUNE_DIR
merge-lora:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR="{{mistral_finetune_dir}}"
    [[ -d "$DIR" ]] || { echo "mistral-finetune not found at $DIR — run: just setup-mistral-finetune"; exit 1; }
    mkdir -p "{{merged_dir}}"
    cd "$DIR"
    .venv/bin/python -m utils.merge_lora \
        --initial_model_ckpt "{{base_model_dir}}/consolidated.safetensors" \
        --lora_ckpt "{{adapter_dir}}/checkpoints/checkpoint_000300/consolidated/lora.safetensors" \
        --dump_ckpt "{{merged_dir}}/consolidated.safetensors" \
        --scaling 2.0
    cp "{{base_model_dir}}/tokenizer.model.v3" "{{merged_dir}}/"
    cp "{{base_model_dir}}/params.json"        "{{merged_dir}}/"
    echo "Merged model ready at {{merged_dir}}"

# One-time: download base model from HF and stage to S3 (run after first infra apply)
stage-model:
    #!/usr/bin/env bash
    set -euo pipefail
    BUCKET=$(cd {{infra_dir}} && tofu output -raw artifacts_bucket)
    hf download mistralai/Mistral-7B-Instruct-v0.3 \
        consolidated.safetensors params.json tokenizer.model.v3 \
        --local-dir /tmp/m7b-v0.3
    aws s3 cp /tmp/m7b-v0.3 \
        "s3://$BUCKET/models/7B-Instruct-v0.3/" \
        --recursive --region {{region}} --profile {{profile}}

# Verify merged artifacts exist in S3 before enabling serve instances
check-merged:
    #!/usr/bin/env bash
    set -euo pipefail
    BUCKET=$(cd {{infra_dir}} && tofu output -raw artifacts_bucket)
    aws s3 ls "s3://$BUCKET/merged/" --region {{region}} --profile {{profile}}

# Backup merged artifacts then destroy all infra (S3 force_destroy=true wipes bucket)
destroy:
    #!/usr/bin/env bash
    set -euo pipefail
    BUCKET=$(cd {{infra_dir}} && tofu output -raw artifacts_bucket)
    echo "Backing up merged artifacts to ./merged-backup/ ..."
    aws s3 sync "s3://$BUCKET/merged/" ./merged-backup/ \
        --region {{region}} --profile {{profile}}
    cd {{infra_dir}} && AWS_PROFILE={{profile}} tofu destroy

# ── training ──────────────────────────────────────────────────────────────────

# Trigger fine-tuning on the train instance via SSM
train:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(cd {{infra_dir}} && tofu output -raw train_ssm_command)"
    echo "Training triggered — run: just train-logs"

# Stream training logs from the train instance (Ctrl-C to exit)
train-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    ID=$(cd {{infra_dir}} && tofu output -raw train_instance_id)
    aws ssm start-session \
        --target "$ID" \
        --document-name "AWS-StartInteractiveCommand" \
        --parameters '{"command": ["tail -f /var/log/pour-train.log"]}' \
        --region {{region}} --profile {{profile}}

# Open a shell on the train instance via SSM
train-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    ID=$(cd {{infra_dir}} && tofu output -raw train_instance_id)
    aws ssm start-session --target "$ID" --region {{region}} --profile {{profile}}

# ── serving ───────────────────────────────────────────────────────────────────

# Serve the merged model locally via vLLM Docker (pre-warms page cache first)
# Requires: Docker with nvidia runtime + merged weights at MERGED_DIR
serve-local:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Pre-warming page cache ..."
    cat "{{merged_dir}}/consolidated.safetensors" > /dev/null
    docker run -d --runtime nvidia --gpus all --ipc=host -p 8000:8000 \
        -v "{{merged_dir}}:/models/m" vllm/vllm-openai:latest \
        --model /models/m --tokenizer_mode mistral --load_format mistral \
        --config_format mistral --served-model-name cocktail-tuned \
        --max-model-len 8192 --enforce-eager
    echo "vLLM serving cocktail-tuned on :8000"

# Trigger vLLM serving on both serve instances via SSM
serve-start:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(cd {{infra_dir}} && tofu output -raw serve_tuned_ssm_command)"
    eval "$(cd {{infra_dir}} && tofu output -raw serve_base_ssm_command)"
    echo "Serving triggered — run: just serve-logs"

# Stream serving logs from the tuned serve instance (Ctrl-C to exit)
serve-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    ID=$(cd {{infra_dir}} && tofu output -raw serve_tuned_instance_id)
    aws ssm start-session \
        --target "$ID" \
        --document-name "AWS-StartInteractiveCommand" \
        --parameters '{"command": ["tail -f /var/log/pour-serve.log"]}' \
        --region {{region}} --profile {{profile}}

# Smoke test: curl /v1/models on the tuned serve instance via SSM send-command
smoke-test:
    #!/usr/bin/env bash
    set -euo pipefail
    ID=$(cd {{infra_dir}} && tofu output -raw serve_tuned_instance_id)
    CMD_ID=$(aws ssm send-command \
        --instance-ids "$ID" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["curl -s localhost:8000/v1/models"]' \
        --region {{region}} --profile {{profile}} \
        --query "Command.CommandId" --output text)
    sleep 8
    aws ssm get-command-invocation \
        --command-id "$CMD_ID" --instance-id "$ID" \
        --region {{region}} --profile {{profile}} \
        --query "StandardOutputContent" --output text

# Open a shell on the tuned serve instance via SSM
serve-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    ID=$(cd {{infra_dir}} && tofu output -raw serve_tuned_instance_id)
    aws ssm start-session --target "$ID" --region {{region}} --profile {{profile}}

# ── eval ──────────────────────────────────────────────────────────────────────

# Run eval on the serve instance via SSM (uses serve_base private IP internally)
eval-remote:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(cd {{infra_dir}} && tofu output -raw eval_ssm_command)"

# Run eval locally against already-running vLLM servers
# Override: BASE_URL / TUNED_URL / BASE_MODEL / TUNED_MODEL via env or args
eval base_url="http://localhost:8001/v1" tuned_url="http://localhost:8000/v1" base_model="cocktail-base" tuned_model="cocktail-tuned":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p reports
    BASE_MODEL={{base_model}} TUNED_MODEL={{tuned_model}} \
    BASE_URL={{base_url}} TUNED_URL={{tuned_url}} \
    uv run python -c "
    import os
    from pathlib import Path
    from pour_decisions.eval_run import run_against_server, render_report
    r = run_against_server(
        Path('data/prepared/test.jsonl'),
        base_model=os.environ['BASE_MODEL'],
        tuned_model=os.environ['TUNED_MODEL'],
        base_url=os.environ['BASE_URL'],
        tuned_url=os.environ['TUNED_URL'],
    )
    report = render_report(r)
    Path('reports/eval-latest.md').write_text(report)
    print(report)
    "

# ── private ───────────────────────────────────────────────────────────────────

# Poll an EC2 instance until /opt/READY exists (bootstrap complete)
_wait-ready id label="instance":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Waiting for bootstrap on {{label}} ({{id}}) ..."
    while true; do
        CMD_ID=$(aws ssm send-command \
            --instance-ids "{{id}}" \
            --document-name "AWS-RunShellScript" \
            --parameters 'commands=["test -f /opt/READY && echo READY || echo WAITING"]' \
            --region {{region}} --profile {{profile}} \
            --query "Command.CommandId" --output text 2>/dev/null) || { sleep 15; continue; }
        sleep 5
        OUT=$(aws ssm get-command-invocation \
            --command-id "$CMD_ID" --instance-id "{{id}}" \
            --region {{region}} --profile {{profile}} \
            --query "StandardOutputContent" --output text 2>/dev/null) || { sleep 10; continue; }
        if [[ "$OUT" == *"READY"* ]]; then
            echo "  {{label}}: bootstrap complete."
            break
        fi
        echo "  {{label}}: still bootstrapping..."
        sleep 15
    done
