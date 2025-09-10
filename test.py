from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
import numpy as np

from src.upt.utils.utils import load_config, resolve_path
from src.upt.utils.loader import ShapeNetCarDataset
from src.upt.models.model import UPTSDF, UPT


def main() -> None:
    # ---------------- Args & config ----------------
    parser = argparse.ArgumentParser(description="Test UPT (ShapeNetCar)")
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
    print(f"[info] device: {device}")

    # ---------------- Data ----------------
    data_cfg: Dict[str, Any] = cfg.get("data", {})
    root_dir = resolve_path(data_cfg.get("root_dir"), base_dir=cfg_dir)
    if root_dir is None:
        raise ValueError("YAML error: data.root_dir must be provided.")
    res = int(data_cfg.get("res", 32))
    mesh_size = int(data_cfg.get("size", 3586))
    sd = bool(data_cfg.get("sd", True))

    # --- TEST indices ---
    test_indices: List[int] = [int(i) for i in data_cfg.get("test_indices", [])]
    if not test_indices:
        raise ValueError("Provide a non-empty list in data.test_indices.")
    bad = [i for i in test_indices if i < 0 or i > 888]
    if bad:
        raise ValueError(f"All data.test_indices must be in [0, 888]. Offending values: {bad}")

    ds = ShapeNetCarDataset(root_dir=root_dir, res=res, size=mesh_size, sd=sd)

    # ---------------- Model ----------------
    model_cfg: Dict[str, Any] = cfg.get("model", {})
    use_sdf = res in (32, 40, 48, 64, 80)

    common_kwargs = dict(
        mesh_pos_embed_dim=int(model_cfg.get("mesh_pos_embed_dim", 768)),
        mesh_pos_include_raw=bool(model_cfg.get("mesh_pos_include_raw", False)),
        mesh_pos_upper=float(model_cfg.get("mesh_pos_upper", 200.0)),
        mesh_num_tokens=int(model_cfg.get("mesh_num_tokens", 1024)),
        mesh_num_heads=int(model_cfg.get("mesh_num_heads", 4)),
        mesh_init_weights=str(model_cfg.get("mesh_init_weights", "truncnormal")),
        appr_depth=int(model_cfg.get("appr_depth", 12)),
        appr_num_heads=int(model_cfg.get("appr_num_heads", 8)),
        appr_drop_path_rate=float(model_cfg.get("appr_drop_path_rate", 0.0)),
        appr_drop_path_decay=bool(model_cfg.get("appr_drop_path_decay", False)),
        appr_init_weights=str(model_cfg.get("appr_init_weights", "truncnormal")),
        appr_init_last_proj_zero=bool(model_cfg.get("appr_init_last_proj_zero", False)),
        dec_num_heads=model_cfg.get("dec_num_heads", None),
        dec_ffn_ratio=int(model_cfg.get("dec_ffn_ratio", 4)),
        dec_dropout=float(model_cfg.get("dec_dropout", 0.0)),
        dec_query_in_dim=int(model_cfg.get("dec_query_in_dim", 3)),
        dec_use_query_mlp=bool(model_cfg.get("dec_use_query_mlp", True)),
        dec_query_mlp_ratio=int(model_cfg.get("dec_query_mlp_ratio", 4)),
    )

    if use_sdf:
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
        model = UPT(**common_kwargs).to(device)
        print("[info] Model: UPT (mesh-only; SDF branch disabled)")

    # ---------------- Losses directory ----------------
    dir = cfg.get("loss_dir", "losses")
    loss_dir = (project_root / str(dir) / "test" / str(res)).resolve()
    loss_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving outputs to: {loss_dir}")

    # ---------------- Load weights ----------------
    out_dir = (project_root / "runs" / f"{res}").resolve()
    loc = "model_weights_" + str(mesh_size) + ".pt"
    ckpt = out_dir / loc
    if not ckpt.exists():
        raise FileNotFoundError(f"Model weights not found at: {ckpt}")
    print(f"[info] loading weights: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])  
    else:
        model.load_state_dict(state)          

    # ---------------- Predictions ----------------
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    test_mse = []

    with torch.no_grad():
        for idx in test_indices:
            mesh, pressure, sdf = ds[idx]                
            mesh = mesh.unsqueeze(0).to(device)          
            pressure = pressure.unsqueeze(0).to(device)  
            if use_sdf:
                sdf = sdf.unsqueeze(0).to(device)        
                pred = model(mesh, sdf)
            else:
                pred = model(mesh)
            loss = criterion(pred, pressure)             
            total_loss += float(loss.item())

    test_mse.append(total_loss / max(len(test_indices), 1))
    print(f"[result] Test MSE: {test_mse[-1]:.6f}  |  #samples={len(test_indices)}")
    ckpt = loss_dir / f"losses_{mesh_size}.csv"
    np.savetxt(ckpt, test_mse, delimiter=",")
    print("[info] Saved:", ckpt)



if __name__ == "__main__":
    main()
