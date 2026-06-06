# 07 — Failures & fixes (the real build log)

Every blocker hit taking Phase 1 from "validated locally" to "training on a live AWS GPU", with root cause and fix. Chronological. Honest. This documents real self-hosted-GPU + fine-tuning + IaC debugging, not just a happy-path demo.

Three themes emerged:
- **AWS GPU capacity & quota are multi-layered and region/AZ-specific** (SRE reality).
- **`templatefile()` + cloud-init escaping is subtle** (IaC depth).
- **`mistral-finetune` has real dependency/runtime friction on a clean modern box** (→ the OSS-contribution track; see spec §14).

---

## A. AWS GPU capacity & quota

### A1 — Spot quota is 0 across all EU regions (and on-demand is 0 in Stockholm)
- **Symptom:** `tofu apply` → `MaxSpotInstanceCountExceeded: Max spot instance count exceeded` on the train instance.
- **Root cause:** "All G and VT **Spot** Instance Requests" (`L-3819A6DF`) is a **separate** quota from "Running On-Demand G and VT instances" (`L-DB2E81BA`), and both are **per-region**. I had checked on-demand in us-east-1/eu-west-1 (768) and wrongly assumed it generalized. Spot G quota was **0 in every EU region checked**; on-demand was **0 in eu-north-1 (Stockholm)** too.
- **Fix:** moved to **eu-central-1 (Frankfurt)** (on-demand G quota 768) and set `use_spot=false`. Spot GPU quota increases are slow/often-denied, so on-demand was the pragmatic unblock.
- **Takeaway:** GPU quota is not one number. Check the **exact quota code, in the exact region, for the exact purchase model** before designing around spot. `aws service-quotas get-service-quota --quota-code L-3819A6DF --region <r>`.

### A2 — g6e.xlarge has no on-demand capacity in eu-central-1
- **Symptom:** `aws_instance.train: Still creating... [5m+]`, while `describe-instances` showed **zero** instances in any state. Tofu was stuck.
- **Root cause:** RunInstances was failing with a **retryable** `InsufficientInstanceCapacity` for g6e.xlarge (L40S — scarce), so the SDK retried silently; no instance ID was ever assigned → nothing to describe.
- **Fix:** switched train to **g5.xlarge** (A10G 24GB — abundant) and dropped `seq_len` 8192→2048 to fit 24GB (records are short JSON, no quality cost).
- **Takeaway:** "Still creating" with no visible instance == masked retryable RunInstances error. Newer/bigger GPU SKUs (L40S) have thin capacity; A10G (g5) is the reliable workhorse.

### A3 — eu-central-1a had no g5 capacity either (AZ-specific)
- **Symptom:** after switching to g5, **same** silent hang.
- **Diagnosis (the useful bit):**
  - `run-instances --dry-run` → *"Request would have succeeded"* → **rules out IAM/SCP/permissions**.
  - real `run-instances` → `InsufficientInstanceCapacity: ... in the Availability Zone you requested (eu-central-1a). You can currently get g5.xlarge capacity by ... choosing eu-central-1b, eu-central-1c.` ← **AWS named the fix**.
- **Root cause:** my single subnet pinned `availability_zones.names[0]` = eu-central-1a, which was GPU-capacity-starved at that moment.
- **Fix:** made the subnet AZ a variable (`var.az_index`, default 1 = eu-central-1b).
- **Takeaway:** GPU capacity varies **by AZ**, not just region. Dry-run isolates permission errors from capacity errors. Don't hardcode AZ[0] for scarce instance types.

### A4 — Region change orphaned a partial stack (cross-region state)
- **Symptom:** after changing `region` and re-planning: `301 PermanentRedirect` / `MovedPermanently` reading the S3 bucket + repo.zip.
- **Root cause:** the first failed apply had already created the bucket/VPC/IAM **in eu-north-1**. Changing the provider region to eu-central-1 left state pointing at resources in another region (S3 is region-bound).
- **Fix:** `tofu destroy -var region=eu-north-1` to cleanly tear down the old-region partial, then `apply` fresh in the new region.
- **Takeaway:** region is not a hot-swap once resources exist. A failed apply leaves real, region-bound resources — clean them in their original region before moving.

### A5 — g5.xlarge (16GB RAM) thrashed loading the model (RAM, not VRAM)
- **Symptom:** training launched, logged up to "Sharding model over 1 GPUs", then **hung ~10 min at 0–1% GPU util**. The python PID was in **`D` (uninterruptible sleep) on `folio_wait_bit_common`** — blocked on page IO, not computing.
- **Root cause:** `g5.xlarge` has only **16GB system RAM**. mistral-finetune loads the full **14.5GB** `consolidated.safetensors` to CPU ("Loaded model on cpu!"), *then* converts to bf16 (a second ~14.5GB allocation), *then* shards to GPU. Peak CPU RAM ≫ 16GB → the mmap'd file fills the page cache (`free -m`: 13.4GB buff/cache, 0 swap) with no room for the conversion copy → constant page eviction + re-read from gp3 EBS → `folio_wait` stall.
- **Fix:** `g5.2xlarge` — **same A10G 24GB GPU, but 32GB RAM** (and 8 vCPU). 32GB absorbs the load + bf16 copy.
- **Takeaway:** GPU instance sizing is **two** budgets — VRAM *and* system RAM. A 7B is ~14.5GB on disk; a load-then-convert path needs ~2× that in CPU RAM transiently. The `D`-state + `folio_wait_bit_common` + 0% GPU signature is "starved on host RAM/IO", not a GPU problem. `nvidia-smi` showing memory-allocated-but-0%-util is the tell.

---

## B. IaC / OpenTofu / cloud-init

### B1 — `templatefile()` escaping broke the bootstrap (`$$(` ≠ `$(`)
- **Symptom:** instance booted, SSM online, but `/opt/READY` never appeared. Bootstrap log: `line 35: syntax error near unexpected token '('`, and `Starting ... at 1351(date ...)`.
- **Root cause:** `templatefile()` only treats `${...}`/`%{...}` as special and collapses `$${` → `${`. A bare **`$$(`** is **not** an escape — it passes through literally, so bash saw `$$` (the PID, e.g. `1351`) followed by `(...)` → syntax error. Command substitutions in a templated script must be plain `$(...)`.
- **Fix:** changed the 5 bootstrap-body `$$(...)` → `$(...)`. (`$${VAR}` for bash `${VAR}` stays correct; brace-less `$VAR`/`$RANDOM` are fine.)
- **Takeaway:** in `templatefile`: `${x}` = interpolate, `$${x}` = literal `${x}`, `$(cmd)` = literal `$(cmd)`, **`$$(cmd)` = broken**. `tofu validate` does **not** render templates with unknown vars, so this only surfaced at boot — a good argument for a render/lint step or a real `plan` smoke.

### B2 — `user_data` edit didn't re-run (replace vs modify)
- **Symptom:** fixed the bootstrap, applied, but the instance kept the old (broken) behavior.
- **Root cause:** `aws_instance.user_data_replace_on_change` defaults to **false**, so a `user_data` change stops/starts the instance and updates the attribute — but **cloud-init runs user_data only once per instance lifetime**, so the new script never executed.
- **Fix:** set `user_data_replace_on_change = true` on both instances; the change then forces a **replacement** (fresh boot → bootstrap runs).
- **Takeaway:** for a bootstrap that must re-run on edit, `user_data_replace_on_change = true` is mandatory. Otherwise you debug a script that isn't the one running.

---

## C. mistral-finetune dependency & runtime friction (→ OSS track)

> All of these are firsthand evidence for the **dependency-modernization PR** recorded in spec §14. The repo's `infra/templates/train-bootstrap.sh.tftpl` now encodes the working install — good before/after material.

### C1 — xformers/torch won't install on Python 3.12
- **Symptom:** `uv pip install -r requirements.txt` → building `xformers==0.0.24` → `ModuleNotFoundError: No module named 'torch'`.
- **Root cause:** the pins (`torch==2.2`, `triton==2.2`, `xformers==0.0.24`, all 2024-era) have **no cp312 wheels**, and the DLAMI defaults to Python 3.12 → uv tried a **source build** of xformers, which imports torch in `setup.py` but doesn't declare it as a PEP-517 build dependency (so build isolation has no torch).
- **Fix:** pin the venv to **Python 3.11** (matching wheels exist) + install `torch` **first** + `uv pip install -r requirements.txt --no-build-isolation` + `numpy<2` (torch 2.2 was built against the NumPy-1.x ABI; numpy 2 triggers `_ARRAY_API not found`).
- **Takeaway:** exact-pinned ML deps rot against new Python. Match the Python to the wheel era; install the build-time dep first; `--no-build-isolation` when a package fails to declare its build deps.

### C2 — `-m utils.*` ModuleNotFoundError (wrong cwd)
- **Symptom:** `Error while finding module specification for 'utils.reformat_data' (No module named 'utils')`.
- **Root cause:** `reformat_data`/`validate_data` were invoked from `/opt/pour-decisions`, but `utils` is a package living in the `mistral-finetune` repo root. `python -m` resolves against `sys.path`/cwd, not the venv.
- **Fix:** run all `-m utils.*` (and `-m train`, `-m utils.merge_lora`) with **`cwd=/opt/mistral-finetune`**; pass data/YAML as **absolute** paths so cwd doesn't affect file locations.
- **Takeaway:** mistral-finetune is run as scripts-from-repo-root, not an installed package. cwd matters for module resolution; absolute paths decouple data location from cwd.

### C3 — `mistral-common` unbounded upper version broke imports
- **Symptom:** `ImportError: cannot import name 'InstructTokenizerBase' from 'mistral_common.tokens.tokenizers.sentencepiece'`.
- **Root cause:** requirements pin `mistral-common>=1.3.1` with **no upper bound**, so a clean install pulled **1.11.2**; `InstructTokenizerBase` was moved out of `sentencepiece` in **1.5+**.
- **Diagnosis:** binary-searched versions on the box → `1.3.1` ✓, `1.4.4` ✓, `1.5.6` ✗.
- **Fix:** pin **`mistral-common==1.4.4`** (newest that still works — honors "use latest within the working bound").
- **Takeaway:** unbounded lower-pins (`>=`) are a time bomb; a transitive lib's refactor silently breaks you. Pin a tested upper bound.

### C4 — `KeyError: 'CUDA_VISIBLE_DEVICES'`
- **Symptom:** torchrun launched, model loaded, then `set_device()` → `KeyError: 'CUDA_VISIBLE_DEVICES'`.
- **Root cause:** `finetune/distributed.py` reads `os.environ['CUDA_VISIBLE_DEVICES']` **directly** (no default). Under systemd (and a non-login SSM shell) that env var isn't set.
- **Fix:** `export CUDA_VISIBLE_DEVICES=0` before torchrun (single GPU).
- **Takeaway:** tools that assume an interactively-set CUDA env break under systemd/CI. Set GPU env explicitly in headless contexts.

---

## D. Serving & eval (two-box, vLLM)

### D1 — Two 7B models won't co-fit on one 24GB GPU
- **Symptom (anticipated, by design):** the eval needs base AND tuned served simultaneously; each is a full 7B (~14.5GB weights). Two on one A10G 24GB = ~29GB → OOM.
- **Fix:** split serving into **two g5.2xlarge boxes** (tuned serves merged on :8000, base serves base on :8000), with a **self-referencing SG ingress on :8000** so the eval (run on the tuned box via SSM) reaches the base box over the private network. Eval points `TUNED_URL=localhost`, `BASE_URL=<base private IP>`.
- **Takeaway:** model count × weight size is a hard VRAM budget. The alternative (one base + LoRA adapter via vLLM `--enable-lora`) fits one GPU and is the cleaner long-term path (Phase 2), but two boxes was the fastest route to first numbers with no code change.

### D2 — vLLM load wedged at "shards 0%" — gp3 IOPS-bound mmap after page-cache eviction
- **Symptom:** tuned box's vLLM stuck at `Loading safetensors checkpoint shards: 0%` for 10+ min, GPU 14GB resident but **0% util**; the `EngineCore` PID in **`D` state on `folio_wait_bit_common`**. The base box loaded the same-size file fine. `--enforce-eager` (ruling out torch.compile) did not help.
- **Diagnosis:** `vmstat` showed disk reading at only **~6 MB/s** (`bi`≈6144). The root volume is **gp3 (3000 IOPS / 125 MB/s)** — so not throughput-bound. vLLM **mmaps** the safetensors and faults it in **page-by-page (4KB random reads)** → IOPS-bound: 3000 IOPS × ~2KB ≈ 6 MB/s. The base box was fast because its model was still **warm in page cache** from the S3-sync write; the tuned box (the eval host) ran `uv sync --extra gpu` afterward, which **evicted the model from cache**, so vLLM re-read it cold at the IOPS wall.
- **Fix:** **sequentially pre-warm the page cache before starting vLLM** (`cat model/consolidated.safetensors > /dev/null`) — a sequential read triggers readahead (large IOs) → throughput-bound ~125MB/s, fills cache in ~2-5 min, then vLLM's mmap faults hit RAM. (Alternative/補: bump gp3 IOPS+throughput.)
- **Takeaway:** mmap load over a cold, IOPS-limited volume is death-by-4KB-fault. `D`-state + `folio_wait_bit_common` + low `vmstat bi` = IO-bound, not GPU. A sequential pre-read converts random faults into readahead. Beware that later steps (here `uv sync`) can evict a cache you relied on.

### D3 — eval crashed writing the report: `reports/` not shipped
- **Symptom:** eval ran end-to-end (hit both servers, mlflow logged) then `FileNotFoundError: reports/eval-latest.md`.
- **Root cause:** `archive_file` excludes `**/reports` from repo.zip (it's generated output), so the dir didn't exist on the box; `open('reports/...','w')` has no parent dir.
- **Fix:** `mkdir -p reports` in the Makefile `eval` target (robust regardless of how the repo arrived).
- **Takeaway:** code that writes into a dir must ensure the dir exists — especially when the deploy artifact deliberately strips generated dirs.

### First real eval result (2026-05-31)
After all of the above, the loop produced: **quantity 0.924→1.000, unit 0.924→1.000, ingredient 1.000→1.000, hallucination 0.000, gate PASS.** The fine-tune reaches perfect quantity+unit normalization on held-out gold vs base 92.4%.

