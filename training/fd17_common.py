"""FD17 paths and atomic bookkeeping; no historical campaign writers."""
# Role: project paths and bookkeeping helpers (JSON read and atomic write, SHA256 manifests,
#   source archiving, progress log, time-limit check) for the scripts of one development
#   iteration, named by the file prefix; `populations` returns the validation-fold populations.
# Used by: fd17_heads.py and fd17_scoring.py; training.
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ROOT: the project root, one level above training/. DEST: this iteration's output folder;
# STATE: its state file; LEDGER: a budget file under models/ (not read by code in this package).
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/fd17-20260913'
STATE = ROOT / 'reports/2026-09-13-fd17-state.json'
LEDGER = ROOT / 'models/evening-20260911/budget.json'
# REVIEW: review root from $CUHKX_REVIEW_ROOT (default ./external-review); OWNER: run-owner name
# from $CUHKX_RUN_OWNER.
REVIEW = Path(os.environ.get('CUHKX_REVIEW_ROOT', 'external-review'))
OWNER = os.environ.get('CUHKX_RUN_OWNER', 'cuhkx-submission')
# Make cuhkx (src/), the numbered scripts and the training modules importable.
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'),str(ROOT/'training')]
# Single-threaded BLAS for reproducible numerics; effective only if numpy is imported afterwards.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


# Local time in ISO 8601 with UTC offset, used to timestamp records.
def now():
    return datetime.now().astimezone().isoformat()


# Read a JSON file (a UTF-8 byte-order mark is tolerated).
def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


# Overall time limit in seconds from DEST/amendment.json, after checking that the file is the
# expected amendment: the earlier limit of 43,200 s plus 3,600 s.
def authorized_cap():
    a=j(DEST/'amendment.json')
    assert a['amendment_id']=='FD17-user-plus3600-20260913-v1'
    assert a['old_cap_seconds']==43200 and a['added_seconds']==3600
    assert a['authorized_global_cap']==a['old_cap_seconds']+a['added_seconds']
    return a['authorized_global_cap']


# Atomic JSON write: create the folder, write a process-specific temporary file, fsync it, then
# rename it over the target, so readers never see a partial file; NaN is rejected.
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


# SHA256 hex digest of a file.
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# {path: SHA256} of the given files, duplicates removed, order kept.
def sources(paths):
    return {str(Path(p)): sha(p) for p in dict.fromkeys(map(Path, paths))}


# (clip table, per-fold populations, length weights) from iterative_common.populations, which
# runs only the `populations` function of next_common.py; imported on first call.
def populations():
    from iterative_common import populations as pure_populations
    return pure_populations()


# Append a timestamped section to this iteration's progress log under reports/.
def progress(text):
    with (ROOT/'reports/2026-09-13-fd17-progress.md').open('a', encoding='utf-8') as f:
        f.write(f'\n## {now()}\n\n{text}\n')


# Copy this iteration's files (scripts/*.py and *.ps1, src/cuhkx/*.py, selected by the prefix in
# their names) into DEST/source-archive/<SHA256>/, check each copy's hash, and return
# their {path: SHA256}.
def archive_code():
    paths = list((ROOT/'scripts').glob('*_fd17_*.py')) + list((ROOT/'scripts').glob('*_fd17_*.ps1')) + list((ROOT/'src/cuhkx').glob('fd17_*.py'))
    paths += list((ROOT/'scripts').glob('fd17_*.py')) + list((ROOT/'scripts').glob('fd17_*.ps1'))
    manifest = sources(paths)
    for path, digest in manifest.items():
        target = DEST/'source-archive'/digest/Path(path).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        assert sha(target) == digest
    return manifest


