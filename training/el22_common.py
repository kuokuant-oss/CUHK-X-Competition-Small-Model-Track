"""EL22 independent accounting; EL21 remains frozen."""
# Role: project paths and small bookkeeping helpers (JSON read and atomic write, SHA256
#   manifests, source archiving, progress log, protected-file check) for the scripts of one
#   development iteration; the file-name prefix names that iteration.
# Used by: el22_p1_heads.py (star import); training.
import os,sys,json,hashlib,shutil
from pathlib import Path
from datetime import datetime
# ROOT: the project root, one level above training/. DEST: this iteration's output folder;
# STATE: its state file under reports/.
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'models/el22-20260915'
STATE=ROOT/'reports/2026-09-15-el22-state.json'
# EXTERNAL: a folder under the review root $CUHKX_REVIEW_ROOT (default ./external-review);
# WORK: a `work` folder beside that root; OWNER: run-owner name from $CUHKX_RUN_OWNER.
EXTERNAL=Path(os.environ.get('CUHKX_REVIEW_ROOT', 'external-review'))/'2026-09-14-post-el21-review'
WORK=EXTERNAL.parents[1]/'work'
OWNER=os.environ.get('CUHKX_RUN_OWNER', 'cuhkx-submission')
# Make cuhkx (src/), the numbered scripts and the training modules importable.
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'training')]
# Single-threaded BLAS for reproducible numerics; effective only if numpy is imported afterwards.
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
# Local time in ISO 8601 with UTC offset, used to timestamp records.
def now():return datetime.now().astimezone().isoformat()
# Read a JSON file (a UTF-8 byte-order mark is tolerated).
def j(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
# SHA256 hex digest of a file.
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
# Atomic JSON write: create the folder, write a process-specific temporary file, flush and fsync
# it, then rename it over the target, so readers never see a partial file; NaN is rejected.
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
# {path: SHA256} of the given files, duplicates removed, order kept.
def sources(ps):return {str(Path(p)):sha(p) for p in dict.fromkeys(map(Path,ps))}
# Copy this iteration's files in scripts/ and src/cuhkx/ (same file-name prefix) once each into
# DEST/source-archive/<SHA256>/; returns their {path: SHA256}.
def archive_code():
    d=sources([*(ROOT/'scripts').glob('el22_*.py'),*(ROOT/'src/cuhkx').glob('el22_*.py')])
    for p,h in d.items():
        out=DEST/'source-archive'/h/Path(p).name
        if not out.exists():out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    return d
# Append a timestamped line to this iteration's progress log under reports/.
def progress(msg):
    with (ROOT/'reports/2026-09-15-el22-progress.md').open('a',encoding='utf-8') as f:f.write(f'\n{now()} {msg}\n')
# Assert that every file listed in DEST/adoption.json, and in C0/frozen-manifest.json when that
# file exists, still has its recorded SHA256.
def protected():
    for p,h in j(DEST/'adoption.json')['protected_sha256'].items():assert sha(p)==h,('PROTECTED FILE CHANGED',p)
    frozen=DEST/'C0/frozen-manifest.json'
    if frozen.exists():
        for p,h in j(frozen)['artifacts'].items():assert sha(p)==h,('FROZEN C0 CHANGED',p)
