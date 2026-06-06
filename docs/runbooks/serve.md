# Serve + eval runbook (automated, SSM-driven)

No SSH. All instance work happens via SSM. Run this only after training has completed and uploaded merged artifacts to `s3://<bucket>/merged/`.

All `just` commands run from your laptop unless labelled **Instance**.

## Prerequisites

Merged artifacts present in S3. Verify:

**Local:**
```bash
just check-merged
```

Expected output: `consolidated.safetensors`, `tokenizer.model.v3`, `params.json`.

## 1. Enable the serve instances

**Local:**
```bash
just infra-serve
```

Runs `tofu apply -var serve_enabled=true` (creates two g5.2xlarge A10G spot instances), then polls both until `/opt/READY` exists on each. No manual SSM session needed.

Bootstrap (cloud-init) installs the SSM agent, Docker, pulls the repo from S3, writes `/opt/pour-decisions/serve.sh`, and touches `/opt/READY`.

## 2. Trigger serving

**Local:**
```bash
just serve-start
```

Fires SSM send-commands on both serve instances (from `tofu output serve_tuned_ssm_command` and `serve_base_ssm_command`).

What `serve.sh` does:

1. Syncs `s3://<bucket>/merged/` to `/opt/pour-decisions/serving/merged/`.
2. `cd /opt/pour-decisions/serving && docker compose up -d` -- starts the tuned model on port 8000 (served as `cocktail-tuned`).
3. Starts the base `mistralai/Mistral-7B-Instruct-v0.3` on port 8001 for eval comparison.

Progress logs to `/var/log/pour-serve.log`. To stream:

**Local** (streams `/var/log/pour-serve.log` via SSM; Ctrl-C to exit):
```bash
just serve-logs
```

## 3. Smoke test

**Local** (fires curl on the instance via SSM send-command; prints JSON response):
```bash
just smoke-test
```

Expected: JSON containing `"id": "cocktail-tuned"`.

## 4. Eval

**Local** (runs eval via SSM on the serve instance; base model reached over private IP):
```bash
just eval-remote
```

Or run eval locally if vLLM endpoints are reachable (e.g. with port forwarding):

**Local:**
```bash
just eval
```

Expected: `reports/eval-latest.md` written with tuned >= base on ingredient accuracy.

## 5. Teardown

**Local** (backs up `merged/` to `./merged-backup/` first, then destroys all infra):
```bash
just destroy
```

`tofu destroy` cancels the persistent spot request for the train instance. Without destroy, a stopped persistent request re-queues and restarts the instance when capacity returns.
