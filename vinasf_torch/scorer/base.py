"""Minimal base utilities required by the Vina scoring function."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..core import ATOMTYPE_MAPPING, COVALENT_RADII_DICT, VDW_RADII_DICT


class BaseScoringFunction:
    """Base class shared by scoring functions in :mod:`vinasf_torch`.

    The implementation mirrors the pieces that are required by the Vina scoring
    function in OpenDock.  It expects the *receptor* and *ligand* objects to
    expose the attributes accessed in the methods below (coordinates, atom types
    and torsion information).  The class merely provides helpers to build
    distance matrices that are reused by :class:`~vinasf_torch.VinaSF`.
    """

    def __init__(
        self,
        receptor: Optional[Any] = None,
        ligand: Optional[Any] = None,
        *,
        atomtype_mapping: Optional[Dict[str, Any]] = None,
        covalent_radii_dict: Optional[Dict[str, float]] = None,
        vdw_radii_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        # Ligand attributes
        if ligand is not None:
            self.ligand = ligand
            self.pose_heavy_atoms_coords = ligand.pose_heavy_atoms_coords
            self.lig_heavy_atoms_element = getattr(ligand, "lig_heavy_atoms_element", None)
            self.updated_lig_heavy_atoms_xs_types = getattr(
                ligand, "updated_lig_heavy_atoms_xs_types", []
            )
            self.lig_root_atom_index = getattr(ligand, "root_heavy_atom_index", None)
            self.lig_frame_heavy_atoms_index_list = getattr(
                ligand, "frame_heavy_atoms_index_list", []
            )
            self.lig_torsion_bond_index = getattr(ligand, "torsion_bond_index", [])
            self.lig_intra_interacting_pairs = getattr(ligand, "intra_interacting_pairs", [])
            self.num_of_lig_ha = getattr(ligand, "number_of_heavy_atoms", None)
            self.number_of_poses = len(ligand.pose_heavy_atoms_coords)
        else:
            self.ligand = None
            self.pose_heavy_atoms_coords = None
            self.updated_lig_heavy_atoms_xs_types = []
            self.lig_root_atom_index = None
            self.lig_frame_heavy_atoms_index_list = []
            self.lig_torsion_bond_index = []
            self.lig_intra_interacting_pairs = []
            self.num_of_lig_ha = None
            self.number_of_poses = 0

        # Receptor attributes
        if receptor is not None:
            self.receptor = receptor
            self.rec_heavy_atoms_xyz = receptor.rec_heavy_atoms_xyz
            self.rec_heavy_atoms_xs_types = getattr(
                receptor, "rec_heavy_atoms_xs_types", []
            )
            self.residues_heavy_atoms_pairs = getattr(
                receptor, "residues_heavy_atoms_pairs", []
            )
            self.heavy_atoms_residues_indices = getattr(
                receptor, "heavy_atoms_residues_indices", []
            )
            self.rec_index_to_series_dict = getattr(
                receptor, "rec_index_to_series_dict", {}
            )
            self.num_of_rec_ha = len(self.rec_heavy_atoms_xyz)
        else:
            self.receptor = None
            self.rec_heavy_atoms_xyz = None
            self.rec_heavy_atoms_xs_types = []
            self.residues_heavy_atoms_pairs = []
            self.heavy_atoms_residues_indices = []
            self.rec_index_to_series_dict = {}
            self.num_of_rec_ha = 0

        # Chemistry dictionaries
        self.atomtype_mapping = atomtype_mapping or ATOMTYPE_MAPPING
        self.covalent_radii_dict = covalent_radii_dict or COVALENT_RADII_DICT
        self.vdw_radii_dict = vdw_radii_dict or VDW_RADII_DICT

        # Distance matrices
        self.dist: Optional[torch.Tensor] = None
        self.intra_dist: Optional[torch.Tensor] = None

    def generate_pldist_mtrx(self) -> torch.Tensor:
        """Generate the protein-ligand distance matrix."""

        if self.receptor is None or self.ligand is None:
            raise ValueError(
                "Both receptor and ligand must be provided to compute distances."
            )

        rec_heavy_atoms_xyz = self.rec_heavy_atoms_xyz
        if rec_heavy_atoms_xyz.dim() == 2:
            rec_heavy_atoms_xyz = rec_heavy_atoms_xyz.unsqueeze(0)
        rec_heavy_atoms_xyz = rec_heavy_atoms_xyz.expand(
            len(self.ligand.pose_heavy_atoms_coords), -1, 3
        )

        ligand_coords = self.ligand.pose_heavy_atoms_coords
        dist = -2 * torch.matmul(
            rec_heavy_atoms_xyz, ligand_coords.permute(0, 2, 1)
        )
        dist += torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, rec_heavy_atoms_xyz.size(1), 1)
        dist += torch.sum(ligand_coords ** 2, -1).view(-1, 1, ligand_coords.size(1))
        dist = (dist >= 0) * dist
        self.dist = torch.sqrt(dist)

        return self.dist

    def generate_intra_mtrx(self) -> torch.Tensor:
        """Generate the ligand intra-molecular distance matrix for interacting pairs."""

        if self.ligand is None:
            raise ValueError("Ligand must be provided to compute intra distances.")

        ligand_coords = self.ligand.pose_heavy_atoms_coords
        num_poses = ligand_coords.size(0)
        if not self.lig_intra_interacting_pairs:
            self.intra_dist = ligand_coords.new_zeros((num_poses, 0))
            return self.intra_dist

        pair_indices = torch.tensor(
            self.lig_intra_interacting_pairs,
            dtype=torch.long,
            device=ligand_coords.device,
        )
        atom_i = ligand_coords.index_select(1, pair_indices[:, 0])
        atom_j = ligand_coords.index_select(1, pair_indices[:, 1])
        self.intra_dist = torch.sqrt(torch.sum(torch.square(atom_i - atom_j), dim=-1))
        return self.intra_dist
