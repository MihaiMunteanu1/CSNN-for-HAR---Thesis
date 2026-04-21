#!/usr/bin/env python3
"""
Fuzeaza feature maps salvate de SaveOutput pe conv1 + conv2 + fc1,
apoi antreneaza un LinearSVC cu GridSearchCV pe C.

Formatul binar (vezi src/TensorWriter.cpp):
  Header:
    uint32  magic = 0x234264FF
    uint8   sparse_flag (0 = dense, 1 = sparse)
    uint32  num_tensors
  Per tensor:
    uint8   label_size
    char[]  label
    uint8   n_dims
    uint16[n_dims] dims
    Date:
      dense  -> float32[product(dims)]
      sparse -> pairs (uint32 index, float32 value), terminated by 0xFFFFFFFF

SaveOutput in KTH_v1.cpp e apelat cu sparse=false, deci folosim calea dense.

Usage:
  python fusion_svm.py \
      --exp-dir /path/to/experiment/working/dir \
      [--layers conv1 conv2 fc1] \
      [--c-grid 0.01 0.1 1 10 100]
"""

import argparse
import os
import struct
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


MAGIC = 0x234264FF


def load_tensor_file(path):
    """Returneaza (labels: list[str], features: np.ndarray [N, D])"""
    with open(path, "rb") as f:
        data = f.read()

    off = 0
    magic = struct.unpack_from("<I", data, off)[0]; off += 4
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic:#x}, expected {MAGIC:#x}")

    sparse_flag = struct.unpack_from("<B", data, off)[0]; off += 1
    num_tensors = struct.unpack_from("<I", data, off)[0]; off += 4

    labels = []
    vectors = []

    for _ in range(num_tensors):
        label_size = struct.unpack_from("<B", data, off)[0]; off += 1
        label = data[off:off+label_size].decode("ascii"); off += label_size

        n_dims = struct.unpack_from("<B", data, off)[0]; off += 1
        dims = struct.unpack_from(f"<{n_dims}H", data, off)
        off += 2 * n_dims
        size = int(np.prod(dims))

        if sparse_flag == 0:
            vec = np.frombuffer(data, dtype=np.float32, count=size, offset=off)
            off += 4 * size
            vec = vec.copy()  # disociaza de mmap
        else:
            vec = np.zeros(size, dtype=np.float32)
            while True:
                idx = struct.unpack_from("<I", data, off)[0]; off += 4
                if idx == 0xFFFFFFFF:
                    break
                val = struct.unpack_from("<f", data, off)[0]; off += 4
                vec[idx] = val

        labels.append(label)
        vectors.append(vec)

    X = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
    return labels, X


def load_split(exp_dir, layers, split):
    """split = 'train' sau 'test'. Returneaza (y, X_fused)"""
    parts = []
    labels_ref = None
    for layer in layers:
        path = os.path.join(exp_dir, f"{layer}_{split}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Lipseste {path}")
        labels, X = load_tensor_file(path)
        print(f"  {layer}_{split}: N={len(labels)}, D={X.shape[1]}")
        if labels_ref is None:
            labels_ref = labels
        elif labels != labels_ref:
            raise ValueError(
                f"Label order difera intre straturi pe split {split}. "
                f"Asigura-te ca SaveOutput ruleaza in aceeasi ordine."
            )
        parts.append(X)
    X_fused = np.hstack(parts)
    return labels_ref, X_fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True,
                    help="Directorul unde au fost scrise fisierele conv1_train etc. "
                         "(working dir-ul in care a rulat KTH_v1).")
    ap.add_argument("--layers", nargs="+", default=["conv1", "conv2", "fc1"],
                    help="Straturi de fuzionat (ordine importanta doar pentru log).")
    ap.add_argument("--c-grid", nargs="+", type=float,
                    default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--no-scale", action="store_true",
                    help="Skip StandardScaler (by default apply scaling on fused features).")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    print(f"[*] Load train din {args.exp_dir}")
    y_train_str, X_train = load_split(args.exp_dir, args.layers, "train")
    print(f"[*] Load test din {args.exp_dir}")
    y_test_str, X_test = load_split(args.exp_dir, args.layers, "test")

    classes = sorted(set(y_train_str))
    lab2id = {c: i for i, c in enumerate(classes)}
    y_train = np.array([lab2id[l] for l in y_train_str])
    y_test = np.array([lab2id[l] for l in y_test_str])

    print(f"[*] Clase: {classes}")
    print(f"[*] X_train {X_train.shape}, X_test {X_test.shape}")

    # Evaluare per strat separat (pentru comparatie)
    print("\n=== Baseline per strat ===")
    offset = 0
    for layer in args.layers:
        path = os.path.join(args.exp_dir, f"{layer}_train")
        _, Xl_train = load_tensor_file(path)
        _, Xl_test = load_tensor_file(os.path.join(args.exp_dir, f"{layer}_test"))
        if args.no_scale:
            clf = LinearSVC(C=1.0, max_iter=5000)
        else:
            clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC(C=1.0, max_iter=5000))
        clf.fit(Xl_train, y_train)
        acc = clf.score(Xl_test, y_test)
        print(f"  {layer}: C=1.0, test acc = {acc*100:.2f}%")

    # Grid search pe fuziune
    print("\n=== Fuziune + GridSearch ===")
    base = LinearSVC(max_iter=5000, dual="auto")
    if args.no_scale:
        estimator = base
        param_grid = {"C": args.c_grid}
    else:
        estimator = make_pipeline(StandardScaler(with_mean=False), base)
        param_grid = {"linearsvc__C": args.c_grid}

    grid = GridSearchCV(estimator, param_grid, cv=args.cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"[*] Best params: {grid.best_params_}")
    print(f"[*] Best CV acc: {grid.best_score_*100:.2f}%")
    test_acc = grid.score(X_test, y_test)
    print(f"[*] TEST acc (fused): {test_acc*100:.2f}%")

    # Confusion matrix simplu
    y_pred = grid.predict(X_test)
    print("\n=== Confusion matrix (test) ===")
    print("rows = true, cols = predicted;", classes)
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1
    for i, row in enumerate(cm):
        correct = row[i]
        total = row.sum()
        pct = 100.0 * correct / total if total > 0 else 0
        print(f"  {classes[i]:<15} {row}  (recall {pct:.1f}%)")


if __name__ == "__main__":
    main()
