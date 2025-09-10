from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.upt.utils.utils import load_config, resolve_path
from src.upt.utils.loader import ShapeNetCarDataset
from src.upt.models.model import UPTSDF, UPT


def main() -> None:
    # ---------------- Args & config ----------------
    parser = argparse.ArgumentParser(description="Train UPT (ShapeNetCar)")
    parser.add_argument("--config", type=str, default="yamls/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent                 
    project_root = cfg_dir.parent             
    cfg = load_config(cfg_path)               

    # ---------------- Device ----------------
    dev_str = str(cfg.get("device", "auto")).strip().lower()
    if dev_str == "cpu":
        device = torch.device("cpu")
    elif dev_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(dev_str)  
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # ---------------- Data ----------------
    data_cfg: Dict[str, Any] = cfg.get("data", {})
    root_dir = resolve_path(data_cfg.get("root_dir"), base_dir=cfg_dir) 
    if root_dir is None:
        raise ValueError("YAML error: data.root_dir must be provided.")
    res = int(data_cfg.get("res", 32))
    mesh_size = int(data_cfg.get("size", 3586))
    mesh_seed = int(data_cfg.get("seed", 42))
    sd = bool(data_cfg.get("sd", True))


    ds = ShapeNetCarDataset(root_dir=root_dir, res=res, size = mesh_size, mesh_seed=mesh_seed, sd=sd)
    N_ALL = 889  # {0, ..., 888}

    # ----- Read TEST indices from YAML, derive TRAIN as complement -----
    test_indices: List[int] = data_cfg.get("test_indices", [])
    if not isinstance(test_indices, list) or len(test_indices) == 0:
        raise ValueError("Provide data.test_indices as a non-empty list of integers in the YAML.")
    # normalize to ints
    test_indices = [int(i) for i in test_indices]
    # range checks against {0,...,888}
    if min(test_indices) < 0 or max(test_indices) > 888:
        raise ValueError("All data.test_indices must be in the inclusive range [0, 888].")
    # optional: uniqueness check
    if len(set(test_indices)) != len(test_indices):
        raise ValueError("data.test_indices contains duplicates; please provide unique indices.")

    # Sanity check w.r.t. dataset size
    if len(ds) < N_ALL:
        print(f"[warn] Dataset has len(ds)={len(ds)} < {N_ALL}. "
              "Complement uses {0,...,888} as requested; ensure indices are valid for your dataset.")

    # TRAIN indices 
    all_ids = set(range(N_ALL))
    train_indices = sorted(all_ids.difference(set(test_indices)))

    if len(ds) <= 888:
        # if dataset is smaller/equal than 889 samples, ensure no index is out of range
        if any(i >= len(ds) for i in train_indices + test_indices):
            raise ValueError("Some indices are out of range for the dataset.")
    
    print(f"[info] #train={len(train_indices)}  #test={len(test_indices)}")

    # ---------------- Output directory (project_root/runs/<res>) ----------------
    dir = cfg.get("output_dir", "runs")
    out_dir = (project_root / str(dir) / str(res)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving outputs to: {out_dir}")

    # ---------------- Losses directory (project_root/losses/<res>) ----------------
    dir = cfg.get("loss_dir", "losses")
    loss_dir = (project_root / str(dir) / "train" / str(res)).resolve()
    loss_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving outputs to: {loss_dir}")

    # ---------------- Dataloader ----------------
    tr_cfg = cfg.get("training", {})
    bs = int(tr_cfg.get("batch_size", 32)) 
    nw = int(tr_cfg.get("num_workers", 2))
    pin = (device.type == "cuda") and bool(tr_cfg.get("pin_memory", True))
    print(f"[info] batch size={bs}, mesh size={mesh_size}")

    train_set = Subset(ds, train_indices)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)

    #-----------Model-----------
    model_cfg: Dict[str, Any] = cfg.get("model", {})
    data_cfg: Dict[str, Any] = cfg.get("data", {})

    valid_res = [32, 40, 48, 64, 80]
    use_sdf = isinstance(res, int) and (res in valid_res)

    # Shared kwargs
    common_kwargs = dict(
        # --- Mesh side ---
        mesh_pos_embed_dim=int(model_cfg.get("mesh_pos_embed_dim", 768)),
        mesh_pos_include_raw=bool(model_cfg.get("mesh_pos_include_raw", False)),
        mesh_pos_upper=float(model_cfg.get("mesh_pos_upper", 200.0)),
        mesh_num_tokens=int(model_cfg.get("mesh_num_tokens", 1024)),
        mesh_num_heads=int(model_cfg.get("mesh_num_heads", 4)),
        mesh_init_weights=str(model_cfg.get("mesh_init_weights", "truncnormal")),
        # --- Approximator ---
        appr_depth=int(model_cfg.get("appr_depth", 12)),
        appr_num_heads=int(model_cfg.get("appr_num_heads", 8)),
        appr_drop_path_rate=float(model_cfg.get("appr_drop_path_rate", 0.0)),
        appr_drop_path_decay=bool(model_cfg.get("appr_drop_path_decay", False)),
        appr_init_weights=str(model_cfg.get("appr_init_weights", "truncnormal")),
        appr_init_last_proj_zero=bool(model_cfg.get("appr_init_last_proj_zero", False)),
        # --- Decoder ---
        dec_num_heads=model_cfg.get("dec_num_heads", None),
        dec_ffn_ratio=int(model_cfg.get("dec_ffn_ratio", 4)),
        dec_dropout=float(model_cfg.get("dec_dropout", 0.0)),
        dec_query_in_dim=int(model_cfg.get("dec_query_in_dim", 3)),
        dec_use_query_mlp=bool(model_cfg.get("dec_use_query_mlp", True)),
        dec_query_mlp_ratio=int(model_cfg.get("dec_query_mlp_ratio", 4)),
    )

    if use_sdf:
        # Grid (SDF) kwargs
        grid_kwargs = dict(
            resolution=int(res),
            patch_size=int(model_cfg.get("patch_size", 2)),
            dims=tuple(model_cfg.get("dims", (192, 384, 768))),
            depths=tuple(model_cfg.get("depths", (2, 2, 2))),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            global_response_norm=bool(model_cfg.get("global_response_norm", True)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
            drop_path_decay=bool(model_cfg.get("drop_path_decay", False)),
            add_type_token=model_cfg.get("add_type_token", None),
        )

        model = UPTSDF(**grid_kwargs, **common_kwargs).to(device)
        print(f"[info] Model: UPTSDF (mesh + SDF), grid res = {res}")
    else:
        # Mesh-only
        model = UPT(**common_kwargs).to(device)
        print("[info] Model: UPT (mesh-only; SDF branch disabled)")

    # ---------------- Optimizer ----------------
    opt_cfg: Dict[str, Any] = cfg.get("optimizer", {"name": "sgd", "lr": 5e-3})
    name = str(opt_cfg.get("name", "sgd")).lower()
    lr = float(opt_cfg.get("lr", 5e-3))
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        momentum = float(opt_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # ---------------- Training ----------------
    epochs = int(tr_cfg.get("epochs", 100))
    loss_fn = nn.MSELoss()
    losses = np.zeros((epochs, 2))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        num_batches = len(train_loader)
        last_pct = -1

        for bidx, (mesh, pressure, sdf) in enumerate(train_loader):
            mesh = mesh.to(device)
            pressure = pressure.to(device)
            if use_sdf:
                sdf = sdf.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_sdf:
                pred = model(mesh, sdf)
            else:
                pred = model(mesh)

            loss = loss_fn(pred, pressure)
            loss.backward()
            optimizer.step()

            # progress: percentage of batches processed in this epoch
            pct = int(100 * (bidx + 1) / max(num_batches, 1))
            if pct != last_pct:
                print(f"Epoch {epoch+1}/{epochs} - {pct:3d}% "
                      f"({bidx+1}/{num_batches} batches)", end="\r", flush=True)
                last_pct = pct

            bs_cur = mesh.size(0)
            total_loss += float(loss.detach()) * bs_cur
            total_count += bs_cur

        print()

        avg = total_loss / max(total_count, 1)
        print(f"Epoch {epoch:03d} | train MSE {avg:.6f}")

        #Store loss evolution
        losses[epoch, 0] = epoch
        losses[epoch, 1] = avg 

    # ---------------- Save ----------------
    ckpt = out_dir / f"model_weights_{mesh_size}.pt"
    torch.save(model.state_dict(), ckpt)
    ckpt1 = loss_dir / f"losses_{mesh_size}.csv"
    np.savetxt(ckpt1, losses, delimiter=",")
    print("[info] Saved:", ckpt)


if __name__ == "__main__":
    main()