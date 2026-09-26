# Evidence

Records of the submitted run (Kaggle submission 56254406), kept so that the result can be checked
without rerunning it. No file derived from the test set is included.

| Path | Contents |
|---|---|
| `runs/warm/`, `runs/cold1/`, `runs/cold2/` | One run from existing caches and two runs rebuilt from the raw test folders: `complete.json` (timings, peak memory, decoder summary, output hashes), `verified.json` (checks against the submitted prediction vector), `take-clock-report.json` (take grouping summary) |
| `runs/fallbacks/receipt.json` | Decoder output when filename clocks are hidden in seven ways (none, depth hidden, IR hidden, skeleton only, one clip in seven, two in three, all) |
| `runtime-versions.json` | Package versions of the machine that produced the submission |
| `folds.csv` | Fold (0–3) of each of the 2,931 training clips used for local validation |

All three runs produced the same prediction vector
(`982db37510d0dd2d86a29358dfe3116f65623a5ece537bb91f2d2b8de87f1860`) and the same CSV
(`aef1ed637146b2c8ef6235b092c88b5869dcbd5e09b88f909e6198bd3cff6191`).

Two edits were made for publication. Absolute paths of the original machine were replaced by
`<package>/` (the package root), `<project>/` (the development project) and `<data>/` (the extracted
competition data), and the list of repeat chains was reduced to its count, because take identifiers
encode the recording times of test clips. The `complete_sha256` values in `verified.json` therefore
refer to the files before these edits.
