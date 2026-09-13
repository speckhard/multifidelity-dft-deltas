#!/bin/bash -l
#SBATCH --job-name=delta-hpc-strict-mt-sweep
#SBATCH --partition=gpu
#SBATCH --account=speckhad
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --array=1-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/physik/%u/delta_data_gpu/retraining_2026-09-09/runs/stage3/strict_mtsweep_%A_%a.out

set -euo pipefail

root="${HPC_ROOT:-/home/physik/speckhad/delta_data_gpu/retraining_2026-09-09}"
repo="${HPC_REPO_DIR:-$root/multifidelity-dft-deltas}"
venv="${HPC_VENV_ACTIVATE:-/home/physik/speckhad/codex_gpu_test/venv-cu126/bin/activate}"
output_root="${HPC_OUTPUT_ROOT:-$root/runs/strict_mt_wandb_sweeps}"
run_dir="${output_root}/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$output_root"
source "$venv"
export PYTHONPYCACHEPREFIX="$root/.pycache"
export WANDB_HPC_RUN_DIR="$run_dir"
cd "$repo"

mkdir -p "$run_dir"

# Required runtime secrets and provenance contract.
test -n "${WANDB_API_KEY:-}"
test -n "${WANDB_SWEEP_PATH:-}"
test -n "${DELTA_CODE_COMMIT:-}"

git_status="$(git status --porcelain)"
if [ -n "$git_status" ]; then
  echo "Dirty git working tree; refuse to launch sweep." >&2
  echo "$git_status" >&2
  exit 1
fi

current_commit="$(git rev-parse --verify HEAD)"
if [ "$DELTA_CODE_COMMIT" != "$current_commit" ]; then
  echo "DELTA_CODE_COMMIT ($DELTA_CODE_COMMIT) does not match current HEAD ($current_commit)." >&2
  exit 1
fi

python - <<'PY'
import os
import re
import torch

commit = os.environ["DELTA_CODE_COMMIT"]
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, "expected one CUDA device"
assert "A100" in torch.cuda.get_device_name(0), "non-A100 runtime"
assert re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit), "invalid DELTA_CODE_COMMIT"
PY

python -m wandb agent --count 1 "$WANDB_SWEEP_PATH"
