from __future__ import annotations
from line_profiler import profile

from pathlib import Path
from typing import Any, Dict, Union
import os

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

PathLike = Union[str, os.PathLike]
__all__ = ["load_config", "resolve_path"]

@profile
def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a Python dict (top-level must be a mapping)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML must be a mapping (key: value).")
    return cfg

@profile
def resolve_path(p: PathLike | None, base_dir: Path) -> Path | None:
    """
    Expand ~ and $VARS; if relative, resolve relative to base_dir.
    Returns None if input is None.
    """
    if p is None:
        return None
    q = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return q if q.is_absolute() else (base_dir / q).resolve()
