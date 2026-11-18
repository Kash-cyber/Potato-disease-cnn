"""
potato_cnn.py
=============
Clean, leak-free version of the original notebook pipeline.

What changed vs. the notebook (see README "Issues found and fixed"):
  * train / validation / test are split by FILE PATH, once, stratified by class, with fixed seeds -
    the three sets can never overlap (the notebook split a re-shuffling tf.data pipeline with take/skip).
  * same CNN architecture as the original (183,747 parameters).
  * EarlyStopping + best-model checkpoint instead of a blind 50 epochs.
  * per-class metrics and a confusion matrix, not just overall accuracy.
  * predict() uses the image it is given (the notebook's version silently read a global variable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

IMAGE_SIZE = 256
BATCH_SIZE = 32
SEED = 123
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_images(data_dir):
    """Return (paths, integer labels, class_names) from a folder-per-class directory."""
    data_dir = Path(data_dir)
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    paths, labels = [], []
    for idx, name in enumerate(class_names):
        for f in sorted((data_dir / name).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                paths.append(str(f))
                labels.append(idx)
    return np.array(paths), np.array(labels), class_names


def split_paths(paths, labels, val=0.1, test=0.1, seed=SEED):
    """Stratified, disjoint split of file paths -> dict of (paths, labels) for train/val/test."""
    p_tr, p_rest, y_tr, y_rest = train_test_split(paths, labels, test_size=val + test, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(p_rest, y_rest, test_size=test / (val + test), stratify=y_rest, random_state=seed)
    return {"train": (p_tr, y_tr), "val": (p_va, y_va), "test": (p_te, y_te)}


def _load(path, label, image_size):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    return img, label


def make_dataset(paths, labels, training=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(lambda p, y: _load(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        aug = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2)])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Model (same architecture as the original notebook)
# --------------------------------------------------------------------------- #
def build_model(n_classes=3, image_size=IMAGE_SIZE):
    if image_size < 192:  # six conv(3x3)+pool(2x2) blocks need at least 192 px to keep a positive feature-map size
        raise ValueError("image_size must be >= 192 for this architecture")
    blocks = [32, 64, 64, 64, 64, 64]
    stack = [layers.Input(shape=(image_size, image_size, 3)), layers.Rescaling(1.0 / 255)]
    for f in blocks:
        stack += [layers.Conv2D(f, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2))]
    stack += [layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(n_classes, activation="softmax")]
    model = models.Sequential(stack)
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# Train / evaluate / predict
# --------------------------------------------------------------------------- #
def train(data_dir, out_dir="results", epochs=50, patience=8, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    paths, labels, class_names = list_images(data_dir)
    parts = split_paths(paths, labels)
    sizes = {k: len(v[0]) for k, v in parts.items()}
    print("images:", len(paths), "| split:", sizes, "| classes:", class_names)
    ds = {k: make_dataset(*v, training=(k == "train"), batch_size=batch_size, image_size=image_size) for k, v in parts.items()}

    model = build_model(len(class_names), image_size)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           tf.keras.callbacks.ModelCheckpoint(str(out / "best_model.keras"), monitor="val_loss", save_best_only=True)]
    hist = model.fit(ds["train"], validation_data=ds["val"], epochs=epochs, callbacks=cbs, verbose=1)
    pd.DataFrame(hist.history).rename_axis("epoch").reset_index().assign(epoch=lambda d: d.epoch + 1).to_csv(out / "history.csv", index=False)

    metrics = evaluate(model, ds["test"], class_names, out)
    metrics.update({"split_sizes": sizes, "epochs_run": len(hist.history["loss"])})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return model, metrics


def evaluate(model, test_ds, class_names, out_dir=None):
    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true += y.numpy().tolist()
        y_pred += np.argmax(model.predict(x, verbose=0), axis=1).tolist()
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    res = {"accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))), "n_test": len(y_true),
           "per_class": {k: rep[k] for k in class_names}, "confusion_matrix": cm.tolist(), "classes": class_names}
    if out_dir:
        pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]).to_csv(Path(out_dir) / "confusion_matrix.csv")
    return res


def predict(model, img, class_names):
    """img: HxWx3 array (0-255) or file path. Returns (class_name, confidence_percent)."""
    if isinstance(img, (str, Path)):
        img = _load(str(img), 0, model.input_shape[1])[0].numpy()
    arr = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), 0)
    probs = model.predict(arr, verbose=0)[0]
    return class_names[int(np.argmax(probs))], round(float(100 * np.max(probs)), 2)
