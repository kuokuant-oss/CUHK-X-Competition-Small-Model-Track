"""Single source of truth for filesystem paths.

Never hard-code an absolute path anywhere else — read it from ``configs/paths.yaml``
through this module, so the repo stays portable if the data moves to another drive.
"""
# Role: reads configs/paths.yaml and looks up the filesystem paths it names (test data, caches,
#   processed tables).
# Used by: scripts/, training/, el22_runtime_support and models.py; both.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# The package root: the folder that contains src/ (this file is src/cuhkx/paths.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "configs" / "paths.yaml"


# The YAML file is read once per process (cached); later edits to it are not seen. Each value is
# wrapped in Path as written and is not joined to REPO_ROOT: an absolute value is used as is, and
# a relative one resolves against the current working directory, which inference.sh and
# run_inference.sh set to the package root.
@lru_cache(maxsize=1)
def paths() -> dict[str, Path]:
    """Return every configured path as a ``Path``."""
    with CONFIG_FILE.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {key: Path(value) for key, value in raw.items()}


# The delivered configs/paths.yaml defines test_root, sample_submission, cache_root,
# processed_root and train_root; keys needed only for training, such as models_root, are not in
# it and raise KeyError here.
def p(key: str) -> Path:
    """Look up one path by key; fail loudly on a typo rather than silently returning None."""
    table = paths()
    if key not in table:
        raise KeyError(f"unknown path key {key!r}; known keys: {sorted(table)}")
    return table[key]
