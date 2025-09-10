import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="yamls/config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------------- Args & config ----------------
    losses_dir = cfg.get("losses_dir", "losses")

    LOSSES_DIR = Path(losses_dir).expanduser().resolve()
    OUT_BASE   = Path("plots")        
    MAX_EPOCHS = 100                  

    train_dict, test_dict, model_sizes = {}, {}, {}

    # ---- Load CSV ----
    for split in ("train", "test"):
        split_dir = LOSSES_DIR / split
        if not split_dir.is_dir():
            continue

        for sd in sorted((p for p in split_dir.iterdir() if p.is_dir()),
                         key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            arrays = []
            sizes_for_sd = []

            for fp in sorted(sd.glob("*.csv")):
                arr = np.loadtxt(fp, delimiter=",")
                arrays.append(arr)
                if split == "train":
                    m = re.search(r"(\d+)\.csv$", fp.name)
                    if m:
                        sizes_for_sd.append(int(m.group(1)))

            if arrays:
                if split == "train":
                    train_dict[sd.name] = arrays
                    model_sizes[sd.name] = sizes_for_sd
                else:
                    test_dict[sd.name] = arrays

    # ---- TRAIN plots ----
    (OUT_BASE / "train").mkdir(parents=True, exist_ok=True)
    for k in sorted(train_dict.keys(), key=lambda s: int(s) if s.isdigit() else s):
        out_dir = OUT_BASE / "train" / k
        out_dir.mkdir(parents=True, exist_ok=True)

        arrs  = train_dict[k]
        sizes = model_sizes.get(k, [None] * len(arrs))

        for i, a in enumerate(arrs):
            a = np.asarray(a)
            
            if a.ndim == 0:
                y = np.array([float(a)])
            elif a.ndim == 1:
                y = a
            else:
                y = a[:, -1]   

            if y.shape[0] <= 1:
                continue

            n = min(MAX_EPOCHS, y.shape[0] - 1)   
            x = np.arange(2, 2 + n, dtype=int)    
            y_plot = y[1:1 + n]

            plt.figure()
            plt.plot(x, y_plot)
            plt.xlabel("Epochs")
            plt.ylabel("MSE")
            size_lbl = sizes[i] if i < len(sizes) else "?"
            plt.title(f"Train ({k}, {size_lbl})")
            plt.grid(True)
            plt.tight_layout()
            fname = f"train_{k}_{size_lbl if size_lbl is not None else i+1}.png"
            plt.savefig(out_dir / fname, dpi=150)
            plt.close()

    # ---- TEST plots ----
    (OUT_BASE / "test").mkdir(parents=True, exist_ok=True)
    for k in sorted(test_dict.keys(), key=lambda s: int(s) if s.isdigit() else s):
        sizes = model_sizes.get(k, [])
        arrs  = test_dict[k]
        n = min(len(sizes), len(arrs))
        if n == 0:
            continue

        xs, ys = [], []
        for i in range(n):
            a = np.asarray(arrs[i])
            if a.size == 0:
                continue
            if a.ndim == 0:
                mse = float(a)
            elif a.ndim == 1:
                mse = float(a[-1])
            else:
                mse = float(a[-1, -1])
            xs.append(float(sizes[i]))
            ys.append(mse)

        if not xs:
            continue

        order = np.argsort(xs)
        x = np.array(xs)[order]
        y = np.array(ys)[order]

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Mesh size")
        plt.ylabel("MSE")
        plt.title(f"Test ({k})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_BASE / "test" / f"test_{k}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()