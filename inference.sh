#!/usr/bin/env bash
# Role: the command that produced the submission: runs the inference entry point on the test
#   folder named in configs/paths.yaml, rebuilding every cache from the raw clip folders.
# Used by: run_inference.sh (after it has written configs/paths.yaml); inference.
# Stop on the first failing command, on unset variables and on failures inside pipelines.
set -euo pipefail
# Work from the package root: relative paths in configs/paths.yaml resolve against it.
cd "$(dirname "$0")"
# Write no __pycache__ folders into the package.
export PYTHONDONTWRITEBYTECODE=1
# Working folders (cache_root and processed_root in configs/paths.yaml point here by default).
mkdir -p outputs/cache outputs/processed outputs/submissions
# Optional overrides: CUHKX_PYTHON (interpreter), DELIVERABLE (checkpoint), FD_OUTPUT (run folder
# that receives prediction.csv and the run records). --active-readout S1-G selects the delivered
# readout (the script's default names one the checkpoint does not contain); --rebuild computes the
# person windows and the det, miw and det248 caches afresh. The cache folder must be empty and
# the run folder must not hold an earlier run; run_inference.sh clears both first.
"${CUHKX_PYTHON:-python}" -B -u scripts/el25r_p1_repeat_runtime.py --pack "${DELIVERABLE:-weights/model.pt}" --output-dir "${FD_OUTPUT:-outputs/run}" --active-readout S1-G --rebuild
