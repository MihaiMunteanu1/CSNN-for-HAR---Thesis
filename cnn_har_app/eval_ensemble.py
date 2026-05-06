"""
Evaluate an ensemble of trained HAR models on the KTH test split.

Usage:
    python3 eval_ensemble.py \
        --checkpoints models/har_conv3d_s0.pth models/har_conv3d_s1.pth ... \
        --data_path ../hog/hog_person_data_new_7.json

Or use a glob pattern:
    python3 eval_ensemble.py --checkpoints "models/har_conv3d_s*.pth"
"""

import argparse
import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HOGDataset, KTH_CLASSES
from model import build_model


def expand_globs(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if matched:
            out.extend(matched)
        elif os.path.exists(p):
            out.append(p)
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../hog/hog_person_data_new_7.json")
    parser.add_argument("--model_type", type=str, default="conv3d",
                        choices=["mlp", "cnn", "temporal", "conv3d"])
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths or glob patterns to .pth checkpoints to ensemble.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--include_diff", action="store_true", default=False)
    parser.add_argument("--include_bbox", action="store_true", default=True)
    parser.add_argument("--no_include_bbox", dest="include_bbox", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpts = expand_globs(args.checkpoints)
    if not ckpts:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints}")

    use_image = args.model_type in ("cnn", "temporal", "conv3d")
    test_dataset = HOGDataset(
        args.data_path, split="test",
        as_image=use_image, augment=False,
        include_diff=args.include_diff and use_image,
        include_bbox=args.include_bbox and use_image,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    sample_x, _ = test_dataset[0]
    sample_shape = tuple(sample_x.shape)
    print(f"Test samples: {len(test_dataset)} | sample shape: {sample_shape}")

    print(f"\nLoading {len(ckpts)} checkpoint(s):")
    models = []
    for ckpt in ckpts:
        m = build_model(hog_shape=sample_shape, model_type=args.model_type)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device).eval()
        models.append(m)
        print(f"  loaded {ckpt}")

    n_classes = len(KTH_CLASSES)
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    individual_correct = [0 for _ in models]
    ensemble_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            probs_sum = None
            for i, m in enumerate(models):
                logits = m(inputs)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
                individual_correct[i] += logits.argmax(1).eq(labels).sum().item()

            avg_probs = probs_sum / len(models)
            preds = avg_probs.argmax(1)
            ensemble_correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf[t, p] += 1

    print("\nIndividual model accuracies:")
    for i, ckpt in enumerate(ckpts):
        acc = 100.0 * individual_correct[i] / max(total, 1)
        print(f"  [{i}] {os.path.basename(ckpt):40s}  {acc:.2f}%")

    ens_acc = 100.0 * ensemble_correct / max(total, 1)
    delta = ens_acc - max(100.0 * c / max(total, 1) for c in individual_correct)
    print(f"\nEnsemble (mean softmax): {ens_acc:.2f}%   (Δ over best single: {delta:+.2f}%)")

    print("\nEnsemble confusion matrix (rows=true, cols=pred):")
    header = "          " + " ".join(f"{c[:5]:>6s}" for c in KTH_CLASSES)
    print(header)
    for i, cls in enumerate(KTH_CLASSES):
        row = " ".join(f"{conf[i, j]:>6d}" for j in range(n_classes))
        recall = 100.0 * conf[i, i] / max(conf[i].sum(), 1)
        print(f"{cls[:9]:>9s} {row}   recall={recall:5.1f}%")


if __name__ == "__main__":
    main()