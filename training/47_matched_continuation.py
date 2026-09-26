"""A0/A1 paired continuation; one invocation runs only explicitly requested fold/arm jobs.

Uses the original trainer, regularization and four-pass inference. Sources are read-only;
each arm gets an independent checkpoint, recipe identity and complete probabilities.
"""

# Role: Helpers shared by the training entry points (assert_idle, sha256, write_json), and main(),
#   the per-fold validation of the 10-epoch continuation recipe that members B and C reuse.
# Used by: 101_matched_replay, 103_b_final_student, 108_finalize_calibrated and
#   124_c_full_student import the helpers; main() is run by hand; training (main() is validation
#   only and not used by the delivered run).
# Main steps of main(), per requested fold k and variant: 1) load the fold-k model of the 60-epoch
#   run, which did not see fold k; 2) predict fold k with four-pass test-time augmentation;
#   3) continue training on the other three folds (10 epochs, peak learning rate 3e-5 by
#   default); 4) predict fold k again and compare with 48_evaluate_matched.evaluate_fold.
# Variants: A0 repeats epoch 1's frame sampling and augmentation in every epoch; A1 draws them anew
#   each epoch. 124_c_full_student.py reuses the A1 configuration recorded here.
# Produces, under models_root/<--out>/: fold<k>/before.npz; fold<k>/<variant>/ with identity.json,
#   checkpoint.pt, metrics.jsonl, student.pt, after.npz and result.json; paired-evaluation.json.
# Inputs, not included in the repository (TRAINING.md lists them): models_root/<--run>/ (default
#   ig65m-t32-det-60ep) with result.json, oof.npz and folds/fold<k>.pt; the det training cache;
#   folds.parquet, takes_train.parquet and takes_test.parquet under processed_root. models_root is
#   not defined in the shipped configs/paths.yaml, and a training run records git rev-parse HEAD,
#   so it needs a git checkout.
# Each variant's identity.json records the SHA256 of the source files, and a resumed run must
#   match it; without the inputs above, main() documents the procedure rather than running out of
#   the box.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# cuhkx lives in src/ and the preprocessing modules in scripts/. The training modules (09, 44, 48)
# are found through the script's own folder, which Python puts on sys.path, or through the path
# set by the importing entry point.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
# Deterministic cuBLAS kernels need this workspace setting before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def assert_idle():
    """Before torch import: no other Python workload, with identity checked ancestors."""
    # Stricter than the pattern check of 09_assert_no_training.py: any other Python process
    # refuses the start, whether it is training or not.
    import psutil

    current = psutil.Process()
    ignored = {current.pid}
    # A Python ancestor is accepted only if it runs with the same arguments; any other Python
    # ancestor refuses the start.
    # Windows venv python.exe is a launcher parent of the real interpreter.
    for ancestor in current.parents():
        if ancestor.name().lower().startswith("python"):
            if ancestor.cmdline()[1:] != current.cmdline()[1:]:
                raise RuntimeError(f"unexpected Python ancestor {ancestor.pid}")
            ignored.add(ancestor.pid)
    # Every Python process except the CPU-only job that work/cpu-coexistence.json may allow.
    processes = import_module("09_assert_no_training")._psutil_processes()
    other = [p for p in processes if p["ProcessId"] not in ignored]
    if other:
        raise RuntimeError(f"REFUSED: competing Python process(es): {other}")
    print(f"exclusive preflight: pid={current.pid}, created={current.create_time()}", flush=True)


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    # Indented JSON; values that JSON cannot encode, such as paths, are written with str().
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ig65m-t32-det-60ep")
    ap.add_argument("--out", default="abc-a-20260910")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--arms", nargs="+", choices=["A0", "A1"], default=["A0", "A1"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert_idle()

    import numpy as np
    import pandas as pd
    import torch

    from cuhkx.fused import FusedFrameDataset, labels_by_row, load_fused, make_fused_transform
    from cuhkx.paths import p
    from cuhkx.soup import held_out_fold
    from cuhkx.train import TrainConfig, _atomic_save, seed_everything, train_one_fold
    from cuhkx.tta import predict

    helper = import_module("44_self_training")
    evaluator = import_module("48_evaluate_matched")
    # Step 1: the 60-epoch run's configuration and its training cache (memory-mapped).
    source_dir = p("models_root") / args.run
    output = p("models_root") / args.out
    if source_dir.resolve() == output.resolve():
        raise ValueError("source and output must differ")
    cfg = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))["config"]
    cfg["extra"].pop("subjects", None)
    extra = cfg["extra"]
    cache_path = p("cache_root") / extra["cache"]
    # Never build a cache implicitly in an experiment launcher.
    if not all(
        cache_path.with_name(cache_path.stem + suffix).exists()
        for suffix in ("_data.npy", "_meta.npz")
    ):
        raise FileNotFoundError(f"pre-existing memmap required: {cache_path}")
    data, offsets, ids, _ = load_fused(cache_path)
    assert isinstance(data, np.memmap)
    # Labelled clips in cache order (order holds their cache rows) with action, fold and subject;
    # row_labels is indexed by cache row.
    table = pd.read_parquet(p("processed_root") / "folds.parquet").set_index("clip_id")
    order = np.array([i for i, name in enumerate(ids) if name in table.index])
    names = np.array([ids[i] for i in order])
    labels = table.loc[names, "action_id"].to_numpy()
    folds = table.loc[names, "fold"].to_numpy()
    subjects = table.loc[names, "user"].to_numpy()
    row_labels = labels_by_row(len(ids), order, labels)
    # Training augmentation as recorded in the run's configuration (random crop, flip, erasing,
    # optional scale jitter); evaluation uses the centre crop.
    train_tf = make_fused_transform(
        extra["crop"],
        erase_prob=extra["erase_prob"],
        flip_prob=extra["flip_prob"],
        scale_range=extra.get("scale_jitter"),
    )
    eval_tf = make_fused_transform(extra["crop"], erase_prob=0, flip_prob=0)

    output.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        # Step 2, per fold k: folds/fold<k>.pt is the model that held fold k out (held_out_fold
        # reads this from the file name), and fold k must share no subject with the other folds.
        source = source_dir / "folds" / f"fold{fold}.pt"
        if held_out_fold(source) != fold or not source.is_file():
            raise ValueError(f"invalid fold source {source}")
        mask = folds == fold
        if set(subjects[mask]) & set(subjects[~mask]):
            raise ValueError("SUBJECT OVERLAP")
        source_hash = sha256(source)
        for arm in args.arms:
            dest = output / f"fold{fold}" / arm
            if dest.exists() and not args.resume:
                raise FileExistsError(f"refusing to overwrite {dest}; use --resume explicitly")
            # Step 3, per variant: the 60-epoch configuration with --epochs and --lr, a log line
            # every epoch, no clean-training subsample in the log, and one weights-only snapshot
            # at the last epoch. augmentation_epoch is only a record; fixed_epoch below acts.
            config = TrainConfig(**cfg)
            config.epochs, config.lr = args.epochs, args.lr
            config.fold = fold
            config.checkpoint_path = dest / "checkpoint.pt"
            config.metrics_path = dest / "metrics.jsonl"
            config.log_every = 1
            config.log_sample = 0
            config.keep_every = args.epochs
            config.extra = dict(extra, augmentation_epoch="fixed1" if arm == "A0" else "1..N")
            # The identity record names the source weights, variant, configuration, held-out
            # subjects and evaluation settings. Its hash is stored in the checkpoint, and
            # train_one_fold refuses to resume a checkpoint with a different hash.
            identity = {
                "source_sha256": source_hash,
                "source": str(source),
                "arm": arm,
                "config": asdict(config),
                "validation_subjects": sorted(set(subjects[mask])),
                "tta": True,
                "estimator": "contiguous/tw=.75/repeats20/seed0",
            }
            config.recipe_hash = hashlib.sha256(
                json.dumps(identity, sort_keys=True, default=str).encode()
            ).hexdigest()
            if args.dry_run:
                print(json.dumps(identity, default=str), flush=True)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            # The git commit and the SHA256 of the source files complete the record. A resumed
            # variant must match the stored record in source, variant, configuration and hashes.
            identity["code_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            identity["code_sha256"] = {
                str(q.relative_to(ROOT)): sha256(q)
                for q in (
                    Path(__file__),
                    ROOT / "src/cuhkx/fused.py",
                    ROOT / "src/cuhkx/train.py",
                    ROOT / "training/48_evaluate_matched.py",
                )
            }
            if (dest / "identity.json").exists():
                old = json.loads((dest / "identity.json").read_text())
                for key in ("source_sha256", "arm", "config", "code_sha256"):
                    if old[key] != json.loads(json.dumps(identity[key], default=str)):
                        raise ValueError(f"resume identity mismatch: {key}")
            else:
                write_json(dest / "identity.json", identity)
            if (dest / "result.json").exists():
                print(f"completed; preserved {dest}", flush=True)
                continue
            # Training: the other three folds with random frame picks and augmentation (A0 reuses
            # the epoch-1 draw in every epoch). Validation: fold k with the segment-centre frames
            # and the centre crop.
            training = FusedFrameDataset(
                data,
                offsets,
                row_labels,
                order[~mask],
                config.n_frames,
                True,
                train_tf,
                config.seed,
                fixed_epoch=1 if arm == "A0" else None,
            )
            validation = FusedFrameDataset(
                data,
                offsets,
                row_labels,
                order[mask],
                config.n_frames,
                False,
                eval_tf,
                config.seed,
            )

            def make_model(source_path=source):
                return helper.load_state(
                    helper.build_model(cfg, data.shape[-1], "cpu"), source_path
                )

            seed_everything(config.seed, deterministic=config.deterministic)
            # Step 4: four-pass predictions of the unchanged fold model on fold k, computed once
            # per fold and shared by both variants.
            before_path = output / f"fold{fold}" / "before.npz"
            if not before_path.exists():
                net = make_model().to(config.device).eval()
                before = predict(net, validation, config.device, config.eval_batch_size, True)
                np.savez(
                    before_path,
                    clip_ids=names[mask],
                    labels=labels[mask],
                    probs=before,
                    folds=folds[mask],
                    source_sha256=source_hash,
                )
                del net
                torch.cuda.empty_cache()
            else:
                blob = np.load(before_path)
                assert str(blob["source_sha256"]) == source_hash
                assert np.array_equal(blob["clip_ids"], names[mask])
                before = blob["probs"]
            print(f"START {arm} fold{fold} epochs={args.epochs} source={source_hash}", flush=True)
            print(f"CUDA free,total={torch.cuda.mem_get_info()}", flush=True)
            # Step 5: train, save the weights, predict fold k again and compare. train_one_fold
            # returns the model, its single-pass accuracy on fold k, the probabilities and the
            # accuracy on the un-augmented training clips.
            started = time.monotonic()
            model, raw_plain, _, fit = train_one_fold(make_model, training, validation, config)
            training_seconds = time.monotonic() - started
            _atomic_save({k: v.cpu() for k, v in model.state_dict().items()}, dest / "student.pt")
            start_eval = time.monotonic()
            after = predict(model.eval(), validation, config.device, config.eval_batch_size, True)
            # Milliseconds per clip for the four-pass prediction.
            latency = 1000 * (time.monotonic() - start_eval) / len(validation)
            np.savez(
                dest / "after.npz",
                clip_ids=names[mask],
                labels=labels[mask],
                folds=folds[mask],
                probs=after,
            )
            # 20 contiguous thinnings of fold k's takes to the test take sizes, scored by per-clip
            # argmax and by joint decoding with the other folds' transition prior at weight 0.75.
            result = evaluator.evaluate_fold(
                fold, names[mask], {"before": before, arm: after}, reference_run=args.run
            )
            result.update(
                {
                    "training_seconds": training_seconds,
                    "ms_per_clip_tta": latency,
                    "weights_bytes_fp32": (dest / "student.pt").stat().st_size,
                    "deliverable_bytes": None,
                    "raw_plain": raw_plain,
                    "clean_train": fit,
                    "source_sha256": source_hash,
                }
            )
            write_json(dest / "result.json", result)
            print(json.dumps(result, default=str), flush=True)
            del model, training, validation
            torch.cuda.empty_cache()
    # Pool the comparisons of every fold found under the output directory into
    # paired-evaluation.json.
    if not args.dry_run:
        evaluator.summarize(output)


if __name__ == "__main__":
    main()
