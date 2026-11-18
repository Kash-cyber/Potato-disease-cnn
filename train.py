"""python train.py --data-dir path/to/PlantVillage-potato --epochs 50"""
import argparse
from src import potato_cnn as pc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the potato-leaf CNN")
    ap.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--out-dir", default="results/new_run")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    a = ap.parse_args()
    _, m = pc.train(a.data_dir, a.out_dir, a.epochs, a.patience)
    print(f"test accuracy {m['accuracy']:.4f} on {m['n_test']} images")
