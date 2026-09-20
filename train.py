"""Reproducible, measured star-versus-sum experiment; no published scores reused."""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST
from model import FashionStarNet

ROOT = Path(__file__).resolve().parent
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def data_and_splits(out, device):
    # Official repository mirror; torchvision checks published MD5 checksums.
    FashionMNIST.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/"
                           "fashion-mnist/master/data/fashion/"]
    original = FashionMNIST(ROOT / "data", train=True, download=True)
    test = FashionMNIST(ROOT / "data", train=False, download=True)
    rng = np.random.default_rng(2026)
    train_indices, val_indices = [], []
    labels = original.targets.numpy()
    for cls in range(10):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        train_indices.extend(idx[:1200])
        val_indices.extend(idx[1200:1400])
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    assert len(set(train_indices) & set(val_indices)) == 0
    assert np.all(np.bincount(labels[train_indices]) == 1200)
    assert np.all(np.bincount(labels[val_indices]) == 200)
    assert len(test) == 10000 and np.all(np.bincount(test.targets.numpy()) == 1000)
    np.savez(out / "split_indices.npz", train=train_indices, validation=val_indices)
    info = {"split_seed": 2026, "train_count": len(train_indices),
            "validation_count": len(val_indices), "test_count": len(test),
            "unused_original_train_count": 46000,
            "train_per_class": 1200, "validation_per_class": 200,
            "test_per_class": 1000, "disjoint_train_validation": True,
            "indices_hash_definition": "SHA256 of little-endian int64 index arrays",
            "train_indices_sha256": digest(train_indices),
            "validation_indices_sha256": digest(val_indices)}
    raw = ROOT / "data" / "FashionMNIST" / "raw"
    info["download_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(raw.glob("*.gz"))}
    def tensor_pair(images, targets):
        return (images.unsqueeze(1).float().div(255).to(device), targets.to(device))
    pairs = [tensor_pair(original.data[idx], original.targets[idx])
             for idx in (train_indices, val_indices)]
    pairs.append(tensor_pair(test.data, test.targets))
    save_json(out / "data_manifest.json", info)
    return pairs, info


@torch.no_grad()
def evaluate(model, pair, batch, return_predictions=False):
    model.eval()
    x, y = pair
    loss_sum = 0.0
    pred_chunks = []
    for start in range(0, len(y), batch):
        logits = model(x[start:start + batch])
        loss_sum += nn.functional.cross_entropy(logits, y[start:start + batch],
                                                reduction="sum").item()
        pred_chunks.append(logits.argmax(1))
    pred = torch.cat(pred_chunks)
    result = {"loss": loss_sum / len(y), "accuracy": (pred == y).float().mean().item()}
    if return_predictions:
        yp, yt = pred.cpu().numpy(), y.cpu().numpy()
        cm = np.zeros((10, 10), dtype=np.int64)
        np.add.at(cm, (yt, yp), 1)
        result["confusion_matrix"] = cm.tolist()
        result["per_class_recall"] = (cm.diagonal() / cm.sum(axis=1)).tolist()
        result["macro_f1"] = float(np.mean(2 * cm.diagonal() /
                                           np.maximum(cm.sum(0) + cm.sum(1), 1)))
        result["correct"] = int((yp == yt).sum())
    return result


def benchmark(device):
    seed_all(1)
    model = FashionStarNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x, y = torch.rand(128, 1, 28, 28, device=device), torch.arange(128, device=device) % 10
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    sync(device)
    start = time.perf_counter()
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    sync(device)
    result = {"device": str(device), "seconds_per_training_batch": (time.perf_counter() - start) / 8,
              "parameters": sum(p.numel() for p in model.parameters())}
    print(json.dumps(result), flush=True)
    return result


def run(args):
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != "auto" else
                          "mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    if args.benchmark:
        benchmark(device)
        return
    started = time.time()
    pairs, split_info = data_and_splits(out, device)
    train, validation, test = pairs
    metadata = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "platform": platform.platform(), "python": sys.version, "pytorch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__, "device": str(device),
        "processor": platform.processor(), "threads": args.threads,
        "cpu_model": subprocess.getoutput("sysctl -n machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor(),
        "epochs": args.epochs, "seeds": args.seeds, "batch_size": 128,
        "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001,
        "adam_betas": [.9, .999], "adam_eps": 1e-8, "scheduler": None,
        "loss": "cross entropy", "preprocessing": "float32 pixels / 255, no augmentation",
        "initialization": "PyTorch defaults; exact shared initial state for paired operations",
        "checkpoint_selection": "highest validation accuracy; earliest epoch wins ties",
        "deterministic_algorithms_enabled": False,
        "determinism_note": "All RNG seeds and epoch permutations fixed; accelerated backend floating-point kernels may not reproduce bit-for-bit across devices/versions.",
        "test_policy": "Evaluate test once per trained model, after best-validation checkpoint selection",
        "source_reference": "https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py",
        "classes": CLASSES}
    save_json(out / "metadata.json", metadata)
    all_curves, all_results, checks = [], [], []
    batch_size = 128
    for seed in args.seeds:
        seed_all(seed)
        initial_model = FashionStarNet("star")
        initial_state = copy.deepcopy(initial_model.state_dict())
        initial_hash = state_digest(initial_state)
        generator = torch.Generator().manual_seed(seed + 10000)
        permutations = [torch.randperm(len(train[1]), generator=generator).to(device)
                        for _ in range(args.epochs)]
        for operation in ("star", "sum"):
            seed_all(seed)
            model = FashionStarNet(operation)
            model.load_state_dict(initial_state)
            assert state_digest(model.state_dict()) == initial_hash
            model = model.to(device)
            # Validate shape and gradient finiteness before training; restore BN buffers afterwards.
            logits = model(train[0][:8])
            assert tuple(logits.shape) == (8, 10)
            nn.functional.cross_entropy(logits, train[1][:8]).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
                       for p in model.parameters())
            model.load_state_dict(initial_state)
            model.zero_grad(set_to_none=True)
            checks.append({"seed": seed, "operation": operation,
                           "initial_state_sha256": initial_hash,
                           "finite_gradients": True, "output_shape": [8, 10]})
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            best_accuracy, best_epoch, best_state = -1, 0, None
            sync(device)
            model_started = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                model.train()
                loss_sum, correct = 0.0, 0
                epoch_started = time.perf_counter()
                for indices in permutations[epoch - 1].split(batch_size):
                    xb, yb = train[0][indices], train[1][indices]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = nn.functional.cross_entropy(logits, yb)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(yb)
                    correct += (logits.argmax(1) == yb).sum().item()
                valid = evaluate(model, validation, batch_size)
                sync(device)
                row = {"seed": seed, "operation": operation, "epoch": epoch,
                       "train_loss": loss_sum / len(train[1]), "train_accuracy": correct / len(train[1]),
                       "validation_loss": valid["loss"], "validation_accuracy": valid["accuracy"],
                       "epoch_seconds": time.perf_counter() - epoch_started}
                all_curves.append(row)
                print(json.dumps(row), flush=True)
                if valid["accuracy"] > best_accuracy:
                    best_accuracy, best_epoch = valid["accuracy"], epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            sync(device)
            train_seconds = time.perf_counter() - model_started
            model.load_state_dict(best_state)
            test_started = time.perf_counter()
            measured = evaluate(model, test, batch_size, return_predictions=True)
            sync(device)
            result = {"seed": seed, "operation": operation, "best_epoch": best_epoch,
                      "best_validation_accuracy": best_accuracy,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "training_seconds_including_validation": train_seconds,
                      "test_seconds": time.perf_counter() - test_started,
                      "test": measured}
            torch.save(best_state, out / f"{operation}_seed{seed}_best.pt")
            all_results.append(result)
            save_json(out / "results.json", all_results)
            save_json(out / "verification.json", checks)
            with (out / "learning_curves.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(all_curves[0]))
                writer.writeheader()
                writer.writerows(all_curves)
            print("RESULT " + json.dumps(result), flush=True)
    metadata["total_wall_seconds_including_data_loading"] = time.time() - started
    metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(out / "metadata.json", metadata)
    summary = {}
    for operation in ("star", "sum"):
        results = [r for r in all_results if r["operation"] == operation]
        accuracies = np.array([r["test"]["accuracy"] for r in results])
        summary[operation] = {"runs": len(results), "accuracy_mean": float(accuracies.mean()),
                              "accuracy_sample_sd": float(accuracies.std(ddof=1)) if len(results) > 1 else None,
                              "macro_f1_mean": float(np.mean([r["test"]["macro_f1"] for r in results]))}
    differences = [next(r["test"]["accuracy"] for r in all_results if r["seed"] == seed and r["operation"] == "star") -
                   next(r["test"]["accuracy"] for r in all_results if r["seed"] == seed and r["operation"] == "sum")
                   for seed in args.seeds]
    summary["paired_accuracy_difference_star_minus_sum"] = differences
    save_json(out / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", default="results")
    parser.add_argument("--benchmark", action="store_true")
    run(parser.parse_args())
