"""Export measured experiment results as publication-ready PNG/PDF figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
results = json.loads((OUT / "results.json").read_text())
metadata = json.loads((OUT / "metadata.json").read_text())
with (OUT / "learning_curves.csv").open() as f:
    curves = list(csv.DictReader(f))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 13, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#7A838D", "text.color": "#22303C",
                     "axes.labelcolor": "#22303C", "xtick.color": "#46515C",
                     "ytick.color": "#46515C", "savefig.facecolor": "white"})
colors = {"star": "#1B6B93", "sum": "#BE7226"}
styles = {"star": "-", "sum": "--"}
markers = {"star": "o", "sum": "s"}
names = {"star": "Fashion-StarNet", "sum": "Sum ablation"}
fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
for operation in ("star", "sum"):
    rows = [r for r in curves if r["operation"] == operation]
    seeds = sorted({int(r["seed"]) for r in rows})
    epochs = sorted({int(r["epoch"]) for r in rows})
    values = []
    for seed in seeds:
        y = [float(next(r for r in rows if int(r["seed"]) == seed and
                         int(r["epoch"]) == epoch)["validation_accuracy"]) * 100
             for epoch in epochs]
        values.append(y)
        ax.plot(epochs, y, styles[operation], color=colors[operation], alpha=.23, lw=1)
    ax.plot(epochs, np.mean(values, axis=0), styles[operation], color=colors[operation],
            marker=markers[operation], markersize=5, lw=2.1, label=names[operation])
ax.set(title="Validation accuracy across five training epochs",
       xlabel="Epoch", ylabel="Validation accuracy (%)", xticks=epochs)
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.grid(axis="y", color="#E3E7EA", lw=.7)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
fig.savefig(FIG / "validation_accuracy.png", dpi=220)
fig.savefig(FIG / "validation_accuracy.pdf")
plt.close(fig)

selected = next(r for r in results if r["seed"] == 42 and r["operation"] == "star")
cm = np.array(selected["test"]["confusion_matrix"])
assert np.all(cm.sum(axis=1) == 1000)
percent = cm / cm.sum(axis=1, keepdims=True) * 100
classes = metadata["classes"]
short = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
         "Shirt", "Sneaker", "Bag", "Ankle boot"]
fig, ax = plt.subplots(figsize=(7.7, 5.8), layout="constrained")
im = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
ax.set(title="Fashion-StarNet test confusion matrix (seed 42)",
       xlabel="Predicted class", ylabel="True class",
       xticks=np.arange(10), yticks=np.arange(10), xticklabels=short, yticklabels=short)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{percent[i,j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if percent[i,j] > 55 else "#22303C")
bar = fig.colorbar(im, ax=ax, fraction=.044, pad=.035)
bar.set_label("Share of true class (%)")
fig.savefig(FIG / "confusion_seed42_star.png", dpi=220)
fig.savefig(FIG / "confusion_seed42_star.pdf")
plt.close(fig)

flat = []
for result in results:
    flat.append({"operation": result["operation"], "seed": result["seed"],
                 "parameters": result["parameters"], "best_epoch": result["best_epoch"],
                 "validation_accuracy": result["best_validation_accuracy"],
                 "test_accuracy": result["test"]["accuracy"],
                 "test_loss": result["test"]["loss"],
                 "macro_f1": result["test"]["macro_f1"],
                 "train_seconds": result["training_seconds_including_validation"],
                 "test_seconds": result["test_seconds"]})
with (OUT / "metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
with (OUT / "per_class_recall.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["operation", "seed", "class", "test_support", "recall"])
    for result in results:
        for cls, recall in zip(classes, result["test"]["per_class_recall"]):
            writer.writerow([result["operation"], result["seed"], cls, 1000, recall])
print("Exported", FIG, "and metrics.csv/per_class_recall.csv")
