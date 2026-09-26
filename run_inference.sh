#!/usr/bin/env bash
# Sets configs/paths.yaml from the arguments, clears the working directories, runs inference.sh
# unchanged, and copies the result to --output.
#
#   ./run_inference.sh --data_dir <TEST_DIR> --sample_submission <SAMPLE_CSV> [--output submission.csv]
set -euo pipefail
# Role: user-facing wrapper around inference.sh; lines 2-5 above are its --help text.
# Used by: the user (see README.md); inference.
# Work from the package root, so relative paths given as arguments are taken relative to it.
cd "$(dirname "$0")"

# Parse the options; --data_dir and --sample_submission are required.
DATA_DIR=""; SAMPLE=""; OUTPUT="submission.csv"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)          DATA_DIR="$2"; shift 2 ;;
    --sample_submission) SAMPLE="$2";   shift 2 ;;
    --output)            OUTPUT="$2";   shift 2 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$DATA_DIR" || -z "$SAMPLE" ]]; then
  echo "error: --data_dir and --sample_submission are required" >&2; exit 2
fi

# Interpreter and run folder, with the same environment variables and defaults as inference.sh.
PY="${CUHKX_PYTHON:-python}"
RUN_DIR="${FD_OUTPUT:-outputs/run}"

# Write configs/paths.yaml (this also checks that --data_dir directly holds SM_test_* folders).
"$PY" tools/configure_paths.py --data_dir "$DATA_DIR" --sample_submission "$SAMPLE"
# Remove the caches and the previous run folder: the entry point needs an empty cache folder and
# a run folder without an earlier run.
rm -rf outputs/cache "$RUN_DIR"
mkdir -p outputs/cache outputs/processed outputs/submissions

# Run the unchanged inference command.
./inference.sh

# Copy the submission that the entry point wrote into the run folder to --output.
SRC_CSV="$RUN_DIR/prediction.csv"
[[ -f "$SRC_CSV" ]] || { echo "error: $SRC_CSV was not produced" >&2; exit 1; }
cp "$SRC_CSV" "$OUTPUT"
echo
echo "submission written to: $OUTPUT"
# Print the row count, the SHA256 of the CSV file and of the predictions as int64 little-endian
# bytes, next to the expected values of the submitted run.
"$PY" - "$OUTPUT" <<'PYEOF'
import hashlib, sys, csv
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
vec = b''.join(int(r['prediction']).to_bytes(8, 'little', signed=True) for r in rows)
print(f"  rows          : {len(rows)}")
print(f"  csv  sha256   : {hashlib.sha256(open(path,'rb').read()).hexdigest()}")
print(f"  vector sha256 : {hashlib.sha256(vec).hexdigest()}")
print( "  expected csv  : aef1ed637146b2c8ef6235b092c88b5869dcbd5e09b88f909e6198bd3cff6191")
print( "  expected vec  : 982db37510d0dd2d86a29358dfe3116f65623a5ece537bb91f2d2b8de87f1860")
PYEOF
