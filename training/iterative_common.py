"""IR0912 shared paths, atomic JSON and pure, frozen population construction."""
# Role: shared paths and helpers (JSON read and atomic write, SHA256, .npz loading) and
#   `populations`, which builds the validation-fold populations by running only the
#   `populations` function of next_common.py.
# Used by: el20_scoring.py and el21_candidates.py directly, fd17_common.py and fd18_common.py
#   through their own `populations`; training.
import ast
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ROOT: the project root, one level above training/. DEST, STATE: output folder and state file
# of one development iteration; LEDGER: a budget file under models/ (none is read in this file).
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/iterative-20260912'
STATE = ROOT / 'reports/2026-09-12-iterative-state.json'
LEDGER = ROOT / 'models/evening-20260911/budget.json'
# REVIEW: review root from $CUHKX_REVIEW_ROOT (default ./external-review); `populations` reads
# the length weights from a JSON file there.
REVIEW = Path(os.environ.get('CUHKX_REVIEW_ROOT', 'external-review'))
# Make cuhkx (src/), the numbered scripts and the training modules importable.
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'),str(ROOT/'training')]
# Single-threaded BLAS for reproducible numerics; effective only if numpy is imported afterwards.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


# Local time in ISO 8601 with UTC offset, used to timestamp records.
def now():
    return datetime.now().astimezone().isoformat()


# Read a JSON file (a UTF-8 byte-order mark is tolerated).
def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


# Atomic JSON write: create the folder, write a process-specific temporary file, fsync it, then
# rename it over the target, so readers never see a partial file; NaN is rejected.
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# SHA256 hex digest of a file.
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# Load an .npz file (pickled objects refused) into a dict of in-memory arrays.
def z(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# Returns (clip table, {fold: populations and prior}, length weights); see next_common.populations.
def populations():
    # Only the pure population function is compiled; never import old state writers.
    # Importing next_common.py would run its module-level code, and the module also defines
    # functions that write state and report files. Instead the file is parsed with `ast`, the
    # single top-level `def populations` is kept, and that one node is compiled and executed in
    # a fresh namespace. Executing a `def` only defines the function, so nothing else in the
    # file runs. The namespace supplies the module-level names the function uses (ROOT, REVIEW,
    # z, j and the ast module; z and j are this module's versions); its other imports happen
    # inside the function.
    path = ROOT / 'training/next_common.py'
    module = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'populations']
    assert len(nodes) == 1
    scope = dict(ROOT=ROOT, REVIEW=REVIEW, z=z, j=j, ast=ast)
    # The node keeps its original line numbers and the real file name is passed, so tracebacks
    # point into next_common.py. The function is then called and its result returned.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['populations']()


# {path: SHA256} of the given files, duplicates removed, order kept.
def sources(paths):
    return {str(Path(p)): sha(p) for p in dict.fromkeys(map(Path, paths))}


# Assert that every file in a {path: SHA256} manifest still has that hash.
def verify(manifest):
    for path, digest in manifest.items():
        assert sha(path) == digest, f'Source changed: {path}'
