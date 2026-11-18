"""Charts built from the training log captured from the original notebook run (data/training_history.csv)."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

GREEN, RED, INK, GREY = "#2f6b3a", "#b3261e", "#222222", "#9aa0a6"


def load_history(path="data/training_history.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _clean(ax):
    ax.grid(True, color="#e8e8e8", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def plot_history(h: pd.DataFrame, path=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2))
    a.plot(h.epoch, h.accuracy * 100, color=GREEN, lw=2, label="Training")
    a.plot(h.epoch, h.val_accuracy * 100, color=RED, lw=1.8, label="Validation")
    a.axhline(95, color=GREY, ls=":", lw=1)
    e95 = int(h.epoch[(h.accuracy >= .95) & (h.val_accuracy >= .95)].min())
    a.annotate(f"both above 95%\nat epoch {e95}", (e95, 95), xytext=(14, 72), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
    dip = h.loc[h.epoch == 30].iloc[0]
    a.annotate(f"epoch 30 dip\n({dip.val_accuracy*100:.1f}%)", (30, dip.val_accuracy * 100), xytext=(33, 74), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
    a.set_ylim(48, 101); a.set_xlabel("Epoch"); a.set_ylabel("Accuracy (%)")
    a.set_title("Accuracy", loc="left", fontweight="bold"); a.legend(frameon=False, loc="lower right"); _clean(a)
    b.plot(h.epoch, h.loss, color=GREEN, lw=2, label="Training")
    b.plot(h.epoch, h.val_loss, color=RED, lw=1.8, label="Validation")
    be = int(h.epoch[h.val_loss.idxmin()])
    b.scatter([be], [h.val_loss.min()], color=INK, zorder=5)
    b.annotate(f"best val loss {h.val_loss.min():.4f}\n(epoch {be})", (be, h.val_loss.min()), xytext=(be - 3, 0.62), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10, ha="center")
    b.set_xlabel("Epoch"); b.set_ylabel("Loss"); b.set_title("Loss", loc="left", fontweight="bold"); b.legend(frameon=False, loc="upper right"); _clean(b)
    fig.suptitle("CNN training over 50 epochs (2,152 images, 3 classes)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_architecture(path=None):
    """Feature-map funnel: spatial size after each conv/pool block (matches the original model.summary())."""
    size, stages = 256, [("Input", 256, 3)]
    for i, ch in enumerate([32, 64, 64, 64, 64, 64], 1):
        size = (size - 2) // 2
        stages.append((f"Conv{i}+Pool", size, ch))
    fig, ax = plt.subplots(figsize=(13, 4.4))
    x = 0
    for name, s, ch in stages:
        w = 0.55
        ax.add_patch(plt.Rectangle((x, -s / 2), w, s, color=GREEN, alpha=0.25 + 0.7 * ch / 64 if name != "Input" else 0.2, ec=GREEN))
        ax.text(x + w / 2, -s / 2 - 14, f"{name}\n{s}x{s}x{ch}", ha="center", va="top", fontsize=9)
        x += 1.35
    ax.add_patch(plt.Rectangle((x, -12), 0.55, 24, color=RED, alpha=0.8))
    ax.text(x + 0.27, -26, "Flatten 256\nDense 64\nSoftmax 3", ha="center", va="top", fontsize=9)
    ax.set_xlim(-0.3, x + 1); ax.set_ylim(-190, 140); ax.axis("off")
    ax.set_title("Images shrink from 256x256 to 2x2 while features deepen: 183,747 parameters in total", loc="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig); return path
    return fig
