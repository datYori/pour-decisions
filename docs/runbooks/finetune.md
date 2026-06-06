# Fine-tune runbook (automated, SSM-driven)

No SSH. All instance work happens via SSM. The instance type is g5.2xlarge (A10G 24GB).

All `just` commands run from your laptop unless labelled **Instance**.

## Prerequisites

- AWS account
- `hf` CLI logged in with the `mistralai/Mistral-7B-Instruct-v0.3` gated-model licence accepted on Hugging Face.
- `just` installed (`brew install just`)

## 1. Provision

**Local:**
```bash
just infra
```

Runs `tofu init` + `tofu apply`, then polls the train instance until `/opt/READY` exists (bootstrap done). No manual SSM session needed.

Bootstrap (cloud-init, runs once on first boot) installs the SSM agent, uv, AWS CLI, clones [mistral-finetune](https://github.com/mistralai/mistral-finetune), installs its Python deps, writes `/opt/pour-decisions/train.sh`, installs and enables the `pour-train.service` systemd unit, and touches `/opt/READY`. No model is downloaded at boot.

## 2. Stage the model (one-time, after bucket exists)

Requires `hf` CLI and the gated licence.

**Local:**
```bash
just stage-model
```

Downloads `consolidated.safetensors`, `params.json`, and `tokenizer.model.v3` from Hugging Face to `/tmp/m7b-v0.3` and uploads them to `s3://<bucket>/models/7B-Instruct-v0.3/`.

## 3. Trigger training

**Local:**
```bash
just train
```

Fires the SSM send-command (from `tofu output train_ssm_command`) to touch `.train-requested` and start `pour-train.service`.

What `train.sh` does (in order):

1. Syncs the base model from `s3://<bucket>/models/7B-Instruct-v0.3/` to `/opt/pour-decisions/models/7B-Instruct-v0.3/`.
2. Generates `/opt/pour-decisions/finetune/7B.box.yaml` — a copy of the committed `7B.yaml` with all four paths rewritten to absolute so the config is cwd-agnostic.
3. Runs `utils.reformat_data` on `train.jsonl` and `val.jsonl`, then `utils.validate_data` against the box config. Aborts on validation failure.
4. Runs `torchrun --nproc-per-node 1` from `/opt/mistral-finetune` against `7B.box.yaml` (single-GPU LoRA).
5. Discovers the latest checkpoint under `runs/cocktail-lora/checkpoints/` and runs `utils.merge_lora` with `--scaling 2.0`, writing the merged weights to `/opt/pour-decisions/serving/merged/consolidated.safetensors`. Copies `tokenizer.model.v3` and `params.json` alongside.
6. Uploads `merged/` (3 files) and `runs/` (all checkpoints) to S3.
7. Touches `/opt/pour-decisions/.train-done`.

## 4. Watch progress

**Local** (streams `/var/log/pour-train.log` via SSM; Ctrl-C to exit):
```bash
just train-logs
```

Expected: `"2,097,152 out of 7,241,732,096 parameters are finetuned (0.03%)."` then decreasing loss. Final lines confirm upload and touch of `.train-done`.

For a raw shell on the instance:

**Local** (opens an SSM shell):
```bash
just train-shell
```

## 5. Spot reclaim behavior

The train instance uses a persistent/stop spot request. On reclaim, AWS stops the instance and preserves the EBS volume (200 GB, `delete_on_termination = false`). The spot request auto-re-queues; when capacity returns the instance restarts and `pour-train.service` fires automatically on boot.

The service's `ExecStart` checks two guards:

- `.train-requested` must exist (set by `just train`).
- `.train-done` must **not** exist (written on success).

If the instance was stopped mid-run, the job restarts from scratch on the next boot. There is no resume from checkpoint; the full training run repeats. This is expected behavior for now. (TODO: checkpoint resume)

To re-run deliberately after a completed job:

**Instance** (via `just train-shell`):
```bash
rm /opt/pour-decisions/.train-done && systemctl start pour-train.service
```

## 6. Artifacts

| Path | Contents |
|------|----------|
| `s3://<bucket>/merged/` | `consolidated.safetensors`, `tokenizer.model.v3`, `params.json` |
| `s3://<bucket>/runs/` | LoRA checkpoints from `runs/cocktail-lora/` |
