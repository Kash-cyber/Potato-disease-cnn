"""
Audit of the ORIGINAL notebook's train/val/test split.

The original notebook built its splits with take()/skip() on a dataset that reshuffles on every pass, then cached
each split separately. This script re-creates exactly that pipeline on SYNTHETIC images whose pixel values encode a
unique ID (same structure as the real data: 1000 / 1000 / 152 files, batch size 32 = 68 batches, 54/6/8 split), runs
the same iteration order as model.fit() + model.evaluate(), and measures how many "test" images were also seen in
the training set.

    python scripts/audit_split_leakage.py            # ~1 minute, no real data needed
"""
import json, os, shutil, sys, tempfile
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
COUNTS = {"Potato___Early_blight": 1000, "Potato___Late_blight": 1000, "Potato___healthy": 152}
SEED_PAIRS = [(123, 12), (1, 2), (5, 6), (7, 8), (9, 10)]   # (image_dataset_from_directory seed, shuffle seed); first = original


def make_dataset(root):
    i = 0
    for cls, n in COUNTS.items():
        (root / cls).mkdir(parents=True)
        for k in range(n):   # constant-colour 8x8 image; R,G encode the image ID and survive resizing
            Image.fromarray(np.full((8, 8, 3), [i % 256, i // 256, 0], dtype=np.uint8)).save(root / cls / f"{k}.png")
            i += 1


def ids(ds):
    out = []
    for x, _ in ds:
        x = x.numpy()
        out += list((np.round(x[:, 0, 0, 0]) + 256 * np.round(x[:, 0, 0, 1])).astype(int))
    return out


def original_partitions(ds, seed, train_split=0.8, val_split=0.1, shuffle_size=10000):
    n = len(ds)                                         # same logic as the notebook's get_dataset_partitions_tf
    ds = ds.shuffle(shuffle_size, seed=seed)
    tr, va = int(train_split * n), int(val_split * n)
    return ds.take(tr), ds.skip(tr).take(va), ds.skip(tr).skip(va)


def main():
    results = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "data"; make_dataset(root)
        for s1, s2 in SEED_PAIRS:
            ds = tf.keras.preprocessing.image_dataset_from_directory(root, seed=s1, shuffle=True, image_size=(8, 8), batch_size=32, verbose=0)
            tr, va, te = original_partitions(ds, s2)
            tr, va, te = [x.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE) for x in (tr, va, te)]
            train_ids = set(ids(tr)); val_ids = set(ids(va))   # first epoch of model.fit(train_ds, validation_data=val_ds)
            test_ids = set(ids(te))                            # model.evaluate(test_ds)
            r = {"seeds": [s1, s2], "n_batches_total": len(ds), "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
                 "test_in_train": len(test_ids & train_ids) / len(test_ids),
                 "val_in_train": len(val_ids & train_ids) / len(val_ids),
                 "test_seen_in_train_or_val": len(test_ids & (train_ids | val_ids)) / len(test_ids)}
            results.append(r); print(json.dumps(r))
    summary = {"mean_test_in_train": float(np.mean([r["test_in_train"] for r in results])),
               "mean_val_in_train": float(np.mean([r["val_in_train"] for r in results])), "trials": results}
    out = ROOT / "results" / "split_leakage_audit.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2)); print("mean test-in-train: %.1f%%" % (100 * summary["mean_test_in_train"]))


if __name__ == "__main__":
    main()
