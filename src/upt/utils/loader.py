import os
from typing import Tuple, Union, Optional
from line_profiler import profile

import torch
from torch.utils.data import Dataset


class ShapeNetCarDataset(Dataset):
    """
    ShapeNet-Car CFD dataset loader.

    Returns a 3-tuple per item: (mesh_positions, pressure, sdf_volume)

    - mesh_positions : (K, 3) tensor of vertex coordinates (K = 'mesh size')
    - pressure       : (K,)   tensor of pressures at selected vertices
    - sdf_volume     : (R, R, R) tensor of sdf values (filled with zeros if SDFs feature disabled)

    Parameters
    ----------
    root_dir : str
        Root directory pointing to 'data' folder.
    res : Union[int, bool], default 32
        Resolution for SDF volumes. Valid: [32, 40, 48, 64, 80].
        If an invalid value is passed (including False/0), SDFs are disabled:
    size : int, default 3586
        Number of mesh points K to keep per sample.
        Subsampling is uniform at random without replacement.
    mesh_seed : Optional[int], default None
        If provided, makes the subsampling deterministic by fixing the seed.
    sd : bool,  default True
        Standardize target pressure values.
    """

    _valid_res = [32, 40, 48, 64, 80]

    def __init__(
        self,
        root_dir: str,
        res: Union[int, bool] = 32,
        size: int = 3586,
        mesh_seed: Optional[int] = None,
        sd : bool = True,
    ):
        self.root_dir = root_dir
        self.res = res
        self.size = int(size)
        self.mesh_seed = mesh_seed

        # Collect all car sample directories
        car_paths = []
        for param_folder in sorted(os.listdir(root_dir)):
            param_path = os.path.join(root_dir, param_folder)
            if not os.path.isdir(param_path):
                continue
            for car_folder in sorted(os.listdir(param_path)):
                car_path = os.path.join(param_path, car_folder)
                if os.path.isdir(car_path):
                    car_paths.append(car_path)

        if not car_paths:
            raise RuntimeError(f"No car data found in {root_dir}")

        # Load one sample to infer mesh/pressure shapes & dtypes
        mesh1 = torch.load(os.path.join(car_paths[0], "mesh_points.th"))
        pressure1 = torch.load(os.path.join(car_paths[0], "pressure.th"))

        self.N = len(car_paths)
        self.P = int(mesh1.shape[0])

        if not (1 <= self.size <= self.P):
            raise ValueError(f"`size` must be in [1, {self.P}], got {self.size}")

        # Preallocate mesh & pressure (load full first, then subsample)
        self.mesh = torch.zeros((self.N, self.P, 3), dtype=mesh1.dtype)
        self.pressure = torch.zeros((self.N, self.P), dtype=pressure1.dtype)

        # Use SDF only if res is a valid integer in the whitelist
        self._use_sdf = isinstance(self.res, int) and (self.res in self._valid_res)

        # Prepare SDF storage
        if self._use_sdf:
            # Infer SDF shape/dtype from first sample at requested resolution
            sdf1 = torch.load(os.path.join(car_paths[0], f"sdf_res{self.res}.th"))
            self.sdf = torch.zeros((self.N, *sdf1.shape), dtype=sdf1.dtype)
        else:
            # SDF disabled: fixed 32^3 zeros for every sample
            self.sdf = torch.zeros((self.N, 32, 32, 32), dtype=torch.float32)

        # Fill tensors (full load)
        for i, car_path in enumerate(car_paths):
            self.mesh[i] = torch.load(os.path.join(car_path, "mesh_points.th"))
            self.pressure[i] = torch.load(os.path.join(car_path, "pressure.th"))
            if self._use_sdf:
                self.sdf[i] = torch.load(os.path.join(car_path, f"sdf_res{self.res}.th"))
            # else: leave zero SDFs as allocated

        # Subsample K mesh points (same indices for all samples)
        if self.size < self.P:
            g = torch.Generator()
            if self.mesh_seed is not None:
                g.manual_seed(int(self.mesh_seed))
            idx = torch.randperm(self.P, generator=g)[: self.size]  # uniform, no replacement
            # Ensure CPU indexing tensors
            idx = idx.to(dtype=torch.int64, device=self.mesh.device)
            self.mesh = self.mesh[:, idx, :].contiguous()      # (N, K, 3)
            self.pressure = self.pressure[:, idx].contiguous() # (N, K)

        #Standardize pressure values:
        if sd:
            self.pressure = (self.pressure - self.pressure.mean()) / self.pressure.std()

    def __len__(self) -> int:
        return self.N
    @profile
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mesh[idx], self.pressure[idx], self.sdf[idx]

    # Convenience getters (optional)
    def get_mesh(self) -> torch.Tensor:
        return self.mesh

    def get_pressure(self) -> torch.Tensor:
        return self.pressure

    def get_sdf(self) -> torch.Tensor:
        return self.sdf
