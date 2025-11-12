"""Adapters that expose RDKit molecules to :mod:`vinasf_torch`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .utils import ATOMTYPE_MAPPING

# Heuristic mapping from the atomic symbol to AutoDock4 atom types.  The
# resulting type is subsequently converted into the X-Score type through the
# ``ATOMTYPE_MAPPING`` table that ships with the package.  Only a subset of the
# AutoDock4 types is exposed because we are targeting small organic molecules
# that are typically handled by RDKit.
_DEFAULT_AD4_TYPE_MAP: Dict[str, str] = {
    "H": "H",
    "C": "C",
    "A": "A",
    "N": "N",
    "O": "OA",
    "S": "S",
    "P": "P",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
    "Si": "Si",
    "Zn": "Zn",
    "Fe": "Fe",
    "Mg": "Mg",
    "Ca": "Ca",
    "Mn": "Mn",
    "Cu": "Cu",
    "Na": "Na",
    "K": "K",
    "Hg": "Hg",
    "Ni": "Ni",
    "Co": "Co",
    "Cd": "Cd",
    "As": "As",
    "Sr": "Sr",
    "U": "U",
    "Cs": "Cs",
    "Mo": "Mo",
}


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RDKit is required to construct VinaSFTorch adapters from RDKit molecules"
        ) from exc
    return Chem


def _get_conformer_coords(mol: Any, conformer_id: int, atom_indices: Sequence[int]) -> List[List[float]]:
    conformer = mol.GetConformer(conformer_id)
    coords: List[List[float]] = []
    for idx in atom_indices:
        position = conformer.GetAtomPosition(idx)
        coords.append([float(position.x), float(position.y), float(position.z)])
    return coords


def _infer_ad4_type(atom: Any, mapping: Dict[str, str]) -> str:
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic() and symbol == "C":
        # Aromatic carbons are represented by ``A`` in AutoDock4.
        symbol = "A"
    try:
        return mapping[symbol]
    except KeyError as exc:
        raise KeyError(f"Unsupported element for AutoDock4 typing: {symbol!r}") from exc


def _compute_intra_pairs(mol: Any, heavy_atom_indices: Sequence[int]) -> List[List[int]]:
    if len(heavy_atom_indices) < 2:
        return []

    Chem = _require_rdkit()
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    index_map = {atom_idx: i for i, atom_idx in enumerate(heavy_atom_indices)}

    intra_pairs: List[List[int]] = []
    for i, atom_i in enumerate(heavy_atom_indices):
        for atom_j in heavy_atom_indices[i + 1 :]:
            path_length = distance_matrix[atom_i, atom_j]
            if path_length == 0:
                continue
            # Skip 1-2, 1-3 and 1-4 interactions (path length <= 3).
            if path_length > 3:
                intra_pairs.append([index_map[atom_i], index_map[atom_j]])
    return intra_pairs


@dataclass
class RDKitLigandAdapter:
    """Minimal ligand representation backed by an RDKit ``Mol``."""

    pose_heavy_atoms_coords: torch.Tensor
    lig_heavy_atoms_element: List[str]
    updated_lig_heavy_atoms_xs_types: List[str]
    root_heavy_atom_index: List[int]
    frame_heavy_atoms_index_list: List[List[int]]
    torsion_bond_index: List[List[int]]
    intra_interacting_pairs: List[List[int]]
    number_of_heavy_atoms: int
    active_torsion: int = 0
    inactive_torsion: int = 0

    @classmethod
    def from_mol(
        cls,
        mol: Any,
        *,
        conformer_id: int = 0,
        atomtype_mapping: Optional[Dict[str, str]] = None,
        ad4_type_map: Optional[Dict[str, str]] = None,
    ) -> "RDKitLigandAdapter":
        Chem = _require_rdkit()
        if mol.GetNumConformers() == 0:
            raise ValueError("Ligand RDKit Mol must contain at least one conformer.")

        heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        heavy_atom_indices = [atom.GetIdx() for atom in heavy_atoms]

        if not heavy_atom_indices:
            raise ValueError("Ligand must contain at least one heavy atom.")

        ad4_map = dict(_DEFAULT_AD4_TYPE_MAP)
        if ad4_type_map:
            ad4_map.update(ad4_type_map)
        atomtype_mapping = atomtype_mapping or ATOMTYPE_MAPPING

        coords = _get_conformer_coords(mol, conformer_id, heavy_atom_indices)
        pose_coords = torch.tensor([coords], dtype=torch.float32)

        xs_types: List[str] = []
        elements: List[str] = []
        for atom in heavy_atoms:
            ad4_type = _infer_ad4_type(atom, ad4_map)
            try:
                xs_type = atomtype_mapping[ad4_type]
            except KeyError as exc:
                raise KeyError(
                    f"Missing X-Score mapping for AutoDock4 type {ad4_type!r}."
                ) from exc
            xs_types.append(xs_type)
            elements.append(atom.GetSymbol())

        intra_pairs = _compute_intra_pairs(mol, heavy_atom_indices)

        return cls(
            pose_heavy_atoms_coords=pose_coords,
            lig_heavy_atoms_element=elements,
            updated_lig_heavy_atoms_xs_types=xs_types,
            root_heavy_atom_index=list(range(len(heavy_atom_indices))),
            frame_heavy_atoms_index_list=[],
            torsion_bond_index=[],
            intra_interacting_pairs=intra_pairs,
            number_of_heavy_atoms=len(heavy_atom_indices),
        )


@dataclass
class RDKitReceptorAdapter:
    """Minimal receptor representation backed by an RDKit ``Mol``."""

    rec_heavy_atoms_xyz: torch.Tensor
    rec_heavy_atoms_xs_types: List[str]
    residues_heavy_atoms_pairs: List[List[int]]
    heavy_atoms_residues_indices: List[int]
    rec_index_to_series_dict: Dict[int, int]

    @classmethod
    def from_mol(
        cls,
        mol: Any,
        *,
        conformer_id: int = 0,
        atomtype_mapping: Optional[Dict[str, str]] = None,
        ad4_type_map: Optional[Dict[str, str]] = None,
    ) -> "RDKitReceptorAdapter":
        if mol.GetNumConformers() == 0:
            raise ValueError("Receptor RDKit Mol must contain at least one conformer.")

        heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        heavy_atom_indices = [atom.GetIdx() for atom in heavy_atoms]

        ad4_map = dict(_DEFAULT_AD4_TYPE_MAP)
        if ad4_type_map:
            ad4_map.update(ad4_type_map)
        atomtype_mapping = atomtype_mapping or ATOMTYPE_MAPPING

        coords = _get_conformer_coords(mol, conformer_id, heavy_atom_indices)
        xyz_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

        xs_types: List[str] = []
        for atom in heavy_atoms:
            ad4_type = _infer_ad4_type(atom, ad4_map)
            try:
                xs_types.append(atomtype_mapping[ad4_type])
            except KeyError as exc:
                raise KeyError(
                    f"Missing X-Score mapping for AutoDock4 type {ad4_type!r}."
                ) from exc

        return cls(
            rec_heavy_atoms_xyz=xyz_tensor,
            rec_heavy_atoms_xs_types=xs_types,
            residues_heavy_atoms_pairs=[],
            heavy_atoms_residues_indices=[],
            rec_index_to_series_dict={},
        )


def RDKitLigandAdapterFactory(
    mol: Any,
    *,
    conformer_id: int = 0,
    atomtype_mapping: Optional[Dict[str, str]] = None,
    ad4_type_map: Optional[Dict[str, str]] = None,
) -> RDKitLigandAdapter:
    """Return a :class:`RDKitLigandAdapter` constructed from ``mol``."""

    return RDKitLigandAdapter.from_mol(
        mol,
        conformer_id=conformer_id,
        atomtype_mapping=atomtype_mapping,
        ad4_type_map=ad4_type_map,
    )


def RDKitReceptorAdapterFactory(
    mol: Any,
    *,
    conformer_id: int = 0,
    atomtype_mapping: Optional[Dict[str, str]] = None,
    ad4_type_map: Optional[Dict[str, str]] = None,
) -> RDKitReceptorAdapter:
    """Return a :class:`RDKitReceptorAdapter` constructed from ``mol``."""

    return RDKitReceptorAdapter.from_mol(
        mol,
        conformer_id=conformer_id,
        atomtype_mapping=atomtype_mapping,
        ad4_type_map=ad4_type_map,
    )


__all__ = [
    "RDKitLigandAdapter",
    "RDKitReceptorAdapter",
    "RDKitLigandAdapterFactory",
    "RDKitReceptorAdapterFactory",
]

