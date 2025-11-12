"""Utility loaders for Vina scoring data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(name: str) -> Dict[str, Any]:
    with open(_DATA_DIR / name, "r") as handle:
        return json.load(handle)


ATOMTYPE_MAPPING = _load_json("atomtype_mapping.json")
COVALENT_RADII_DICT = _load_json("covalent_radii_dict.json")
VDW_RADII_DICT = _load_json("vdw_radii_dict.json")

__all__ = ["ATOMTYPE_MAPPING", "COVALENT_RADII_DICT", "VDW_RADII_DICT"]
