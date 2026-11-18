"""
Tests use SYNTHETIC images (random colour patches) purely to check the code runs and the split is leak-free.
They say nothing about real model accuracy.   Run: python tests/test_pipeline.py
"""
import os, sys, tempfile
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import potato_cnn as pc


def make_fake(root, per_class=(40, 40, 12), size=64):
    rng = np.random.default_rng(0)
    for name, n, base in zip(["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"], per_class, (40, 120, 200)):
        (root / name).mkdir(parents=True)
        for i in range(n):
            arr = np.clip(rng.normal(base, 25, (size, size, 3)), 0, 255).astype("uint8")
            Image.fromarray(arr).save(root / name / f"{i}.jpg")


def test_all():
    # 1. architecture matches the original notebook exactly
    assert pc.build_model().count_params() == 183_747, "parameter count differs from the original model"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "data"; make_fake(root)
        paths, labels, classes = pc.list_images(root)
        assert classes == ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
        # 2. split is disjoint and stratified
        parts = pc.split_paths(paths, labels)
        sets = {k: set(v[0]) for k, v in parts.items()}
        assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
        assert sum(len(s) for s in sets.values()) == len(paths)
        assert all(2 in parts[k][1] for k in parts), "every split should contain the rare class"
        # 3. end-to-end: train 2 epochs at small size, evaluate, predict
        model, m = pc.train(root, Path(d) / "out", epochs=2, image_size=192, batch_size=16)
        assert (Path(d) / "out" / "history.csv").exists() and (Path(d) / "out" / "confusion_matrix.csv").exists()
        assert 0 <= m["accuracy"] <= 1 and sum(map(sum, m["confusion_matrix"])) == m["n_test"]
        # 4. predict() uses the image it receives
        a = pc.predict(model, str(next((root / classes[0]).iterdir())), classes)
        assert a[0] in classes and 0 <= a[1] <= 100
        try:
            pc.build_model(image_size=64); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("OK: all tests passed")


if __name__ == "__main__":
    test_all()
