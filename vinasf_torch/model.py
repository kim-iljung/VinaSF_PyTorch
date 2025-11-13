from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import torch

from .utils import ATOMTYPE_MAPPING, COVALENT_RADII_DICT, VDW_RADII_DICT


def _like_of(*tensors: Optional[torch.Tensor]) -> torch.Tensor:
    """첫 번째로 발견한 텐서를 기준으로 device/dtype 컨텍스트를 반환.
    모두 None이면 CPU float32 스칼라 텐서를 돌려준다."""
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t
    return torch.zeros((), dtype=torch.float32)


class VinaSFTorch(torch.nn.Module):
    """AutoDock Vina 점수의 PyTorch 구현(가우시안/반발/소수성/H-결합 항 포함)."""

    def __init__(
        self,
        receptor: Optional[Any] = None,
        ligand: Optional[Any] = None,
        *,
        atomtype_mapping: Optional[Dict[str, Any]] = None,
        covalent_radii_dict: Optional[Dict[str, float]] = None,
        vdw_radii_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()

        self._initialize_entities(
            receptor,
            ligand,
            atomtype_mapping=atomtype_mapping,
            covalent_radii_dict=covalent_radii_dict,
            vdw_radii_dict=vdw_radii_dict,
        )

        # 자주 쓰는 버퍼를 등록(자동 이동용). 초기엔 빈 텐서.
        self.register_buffer("dist", torch.empty(0))
        self.register_buffer("intra_dist", torch.empty(0))
        self.register_buffer("intra_repulsive_term", torch.tensor(1e-6))
        self.register_buffer("inter_repulsive_term", torch.tensor(1e-6))
        self.register_buffer("FR_repulsive_term", torch.tensor(1e-6))

        self.repulsive_ = 6
        self.vina_inter_energy: torch.Tensor = torch.tensor(0.0)

        self.all_root_frame_heavy_atoms_index_list = (
            [self.lig_root_atom_index] + self.lig_frame_heavy_atoms_index_list
            if self.lig_root_atom_index is not None
            else self.lig_frame_heavy_atoms_index_list
        )
        self.number_of_all_frames = len(self.all_root_frame_heavy_atoms_index_list)

        self.lig_intra_interacting_pairs = (
            self.ligand.intra_interacting_pairs if self.ligand is not None else []
        )

        # 준비 과정에서 만들어지는 텐서들(동적으로 계산됨)
        self.rec_lig_is_hydrophobic: Optional[torch.Tensor] = None
        self.rec_lig_is_hbond: Optional[torch.Tensor] = None
        self.rec_lig_atom_vdw_sum: Optional[torch.Tensor] = None
        self.vina_dist: Optional[torch.Tensor] = None

        self.intra_rec_lig_is_hydrophobic: Optional[torch.Tensor] = None
        self.intra_rec_lig_is_hbond: Optional[torch.Tensor] = None
        self.intra_rec_lig_atom_vdw_sum: Optional[torch.Tensor] = None
        self.intra_vina_dist: Optional[torch.Tensor] = None

    # -------------------------- 초기화 --------------------------
    def _initialize_entities(
        self,
        receptor: Optional[Any],
        ligand: Optional[Any],
        *,
        atomtype_mapping: Optional[Dict[str, Any]],
        covalent_radii_dict: Optional[Dict[str, float]],
        vdw_radii_dict: Optional[Dict[str, float]],
    ) -> None:
        if ligand is not None:
            self.ligand = ligand
            self.pose_heavy_atoms_coords = ligand.pose_heavy_atoms_coords
            self.lig_heavy_atoms_element = getattr(ligand, "lig_heavy_atoms_element", None)
            self.updated_lig_heavy_atoms_xs_types = getattr(ligand, "updated_lig_heavy_atoms_xs_types", [])
            self.lig_root_atom_index = getattr(ligand, "root_heavy_atom_index", None)
            self.lig_frame_heavy_atoms_index_list = getattr(ligand, "frame_heavy_atoms_index_list", [])
            self.lig_torsion_bond_index = getattr(ligand, "torsion_bond_index", [])
            self.lig_intra_interacting_pairs = getattr(ligand, "intra_interacting_pairs", [])
            self.num_of_lig_ha = getattr(ligand, "number_of_heavy_atoms", None)
            self.number_of_poses = len(ligand.pose_heavy_atoms_coords)
        else:
            self.ligand = None
            self.pose_heavy_atoms_coords = None
            self.lig_heavy_atoms_element = None
            self.updated_lig_heavy_atoms_xs_types = []
            self.lig_root_atom_index = None
            self.lig_frame_heavy_atoms_index_list = []
            self.lig_torsion_bond_index = []
            self.lig_intra_interacting_pairs = []
            self.num_of_lig_ha = None
            self.number_of_poses = 0

        if receptor is not None:
            self.receptor = receptor
            self.rec_heavy_atoms_xyz = receptor.rec_heavy_atoms_xyz
            self.rec_heavy_atoms_xs_types = getattr(receptor, "rec_heavy_atoms_xs_types", [])
            self.residues_heavy_atoms_pairs = getattr(receptor, "residues_heavy_atoms_pairs", [])
            self.heavy_atoms_residues_indices = getattr(receptor, "heavy_atoms_residues_indices", [])
            self.rec_index_to_series_dict = getattr(receptor, "rec_index_to_series_dict", {})
            self.num_of_rec_ha = self.rec_heavy_atoms_xyz.size(-2)
        else:
            self.receptor = None
            self.rec_heavy_atoms_xyz = None
            self.rec_heavy_atoms_xs_types = []
            self.residues_heavy_atoms_pairs = []
            self.heavy_atoms_residues_indices = []
            self.rec_index_to_series_dict = {}
            self.num_of_rec_ha = 0

        self.atomtype_mapping = atomtype_mapping or ATOMTYPE_MAPPING
        self.covalent_radii_dict = covalent_radii_dict or COVALENT_RADII_DICT
        self.vdw_radii_dict = vdw_radii_dict or VDW_RADII_DICT

    # -------------------------- torch.nn.Module override --------------------------
    def to(self, *args: Any, **kwargs: Any) -> "VinaSFTorch":  # type: ignore[override]
        """모듈과 캐시된 텐서들을 요청된 device/dtype로 이동."""
        to_kwargs = self._parse_to_kwargs(*args, **kwargs)
        module = super().to(*args, **kwargs)
        module._move_cached_tensors(**to_kwargs)
        return module

    @staticmethod
    def _parse_to_kwargs(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        device: Optional[torch.device] = kwargs.get("device")
        dtype: Optional[torch.dtype] = kwargs.get("dtype")
        non_blocking: bool = kwargs.get("non_blocking", False)
        copy: bool = kwargs.get("copy", False)
        memory_format = kwargs.get("memory_format")

        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, str):
                device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                dtype = arg
            elif torch.is_tensor(arg):
                device = arg.device
                if dtype is None:
                    dtype = arg.dtype

        to_kwargs: Dict[str, Any] = {}
        if device is not None:
            to_kwargs["device"] = device
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        if non_blocking:
            to_kwargs["non_blocking"] = non_blocking
        if copy:
            to_kwargs["copy"] = copy
        if memory_format is not None:
            to_kwargs["memory_format"] = memory_format

        return to_kwargs

    def _move_cached_tensors(self, **to_kwargs: Any) -> None:
        if not to_kwargs:
            return

        device = to_kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
            to_kwargs = dict(to_kwargs)
            to_kwargs["device"] = device

        def convert(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            return tensor.to(**to_kwargs)

        # 리간드 좌표
        if self.ligand is not None:
            lig_coords = getattr(self.ligand, "pose_heavy_atoms_coords", None)
            if isinstance(lig_coords, torch.Tensor):
                converted = convert(lig_coords)
                if converted is not None:
                    self.ligand.pose_heavy_atoms_coords = converted
                    self.pose_heavy_atoms_coords = converted

        # 수용체 좌표
        if self.receptor is not None:
            rec_coords = getattr(self.receptor, "rec_heavy_atoms_xyz", None)
            if isinstance(rec_coords, torch.Tensor):
                converted = convert(rec_coords)
                if converted is not None:
                    self.receptor.rec_heavy_atoms_xyz = converted
                    self.rec_heavy_atoms_xyz = converted

        # 모듈 속성으로 저장된 텐서들
        for attr in (
            "dist",
            "intra_dist",
            "intra_repulsive_term",
            "inter_repulsive_term",
            "FR_repulsive_term",
            "vina_inter_energy",
            "rec_lig_is_hydrophobic",
            "rec_lig_is_hbond",
            "rec_lig_atom_vdw_sum",
            "vina_dist",
            "intra_rec_lig_is_hydrophobic",
            "intra_rec_lig_is_hbond",
            "intra_rec_lig_atom_vdw_sum",
            "intra_vina_dist",
        ):
            tensor = getattr(self, attr, None)
            if isinstance(tensor, torch.Tensor):
                converted = convert(tensor)
                if converted is not None:
                    setattr(self, attr, converted)

    # -------------------------- RDKit 어댑터 --------------------------
    @classmethod
    def from_rdkit(
        cls,
        receptor_mol: Any,
        ligand_mol: Any,
        *,
        receptor_conformer_id: int = 0,
        ligand_conformer_id: int = 0,
        atomtype_mapping: Optional[Dict[str, Any]] = None,
        covalent_radii_dict: Optional[Dict[str, float]] = None,
        vdw_radii_dict: Optional[Dict[str, float]] = None,
    ) -> "VinaSFTorch":
        from .rdkit_adapter import RDKitLigandAdapter, RDKitReceptorAdapter

        receptor = RDKitReceptorAdapter.from_mol(
            receptor_mol,
            conformer_id=receptor_conformer_id,
            atomtype_mapping=atomtype_mapping,
        )
        ligand = RDKitLigandAdapter.from_mol(
            ligand_mol,
            conformer_id=ligand_conformer_id,
            atomtype_mapping=atomtype_mapping,
        )

        return cls(
            receptor=receptor,
            ligand=ligand,
            atomtype_mapping=atomtype_mapping,
            covalent_radii_dict=covalent_radii_dict,
            vdw_radii_dict=vdw_radii_dict,
        )

    # -------------------------- 거리 행렬 생성 --------------------------
    def generate_pldist_mtrx(self) -> torch.Tensor:
        if self.receptor is None or self.ligand is None:
            raise ValueError("Both receptor and ligand must be provided to compute distances.")

        rec = self.rec_heavy_atoms_xyz
        lig = self.ligand.pose_heavy_atoms_coords

        if isinstance(rec, torch.Tensor):
            if rec.device != lig.device or rec.dtype != lig.dtype:
                rec = rec.to(device=lig.device, dtype=lig.dtype)
                self.receptor.rec_heavy_atoms_xyz = rec
                self.rec_heavy_atoms_xyz = rec

        rec = rec.expand(len(lig), -1, -1)  # (Nposes, Nrec, 3)

        dist = -2 * torch.matmul(rec, lig.permute(0, 2, 1))
        dist += torch.sum(rec ** 2, -1).unsqueeze(-1)
        dist += torch.sum(lig ** 2, -1).unsqueeze(1)
        dist = torch.clamp_min(dist, 0)
        self.dist = torch.sqrt(dist)

        return self.dist

    def generate_intra_mtrx(self) -> torch.Tensor:
        if self.ligand is None:
            raise ValueError("Ligand must be provided to compute intra distances.")

        lig = self.ligand.pose_heavy_atoms_coords  # (Nposes, Alig, 3)
        num_poses = lig.size(0)

        if not self.lig_intra_interacting_pairs:
            self.intra_dist = lig.new_zeros((num_poses, 0))
            return self.intra_dist

        pair_indices = torch.as_tensor(
            self.lig_intra_interacting_pairs, dtype=torch.long, device=lig.device
        )  # (K, 2)
        atom_i = lig.index_select(1, pair_indices[:, 0])
        atom_j = lig.index_select(1, pair_indices[:, 1])
        self.intra_dist = torch.sqrt(torch.sum((atom_i - atom_j) ** 2, dim=-1))
        return self.intra_dist

    # -------------------------- 보조 기능 --------------------------
    @staticmethod
    def _pad(vector: torch.Tensor, max_len: int) -> torch.Tensor:
        """벡터 뒤에 0을 붙여 길이를 max_len으로 맞춤."""
        pad = max_len - vector.numel()
        if pad <= 0:
            return vector
        return torch.cat([vector, vector.new_zeros(pad)], dim=0)

    # -------------------------- 데이터 준비(상호작용) --------------------------
    def _prepare_data(self) -> "VinaSFTorch":
        device_dtype_ref = _like_of(self.dist, self.pose_heavy_atoms_coords, self.rec_heavy_atoms_xyz)

        rec_atom_indices_list = []
        lig_atom_indices_list = []
        all_selected_rec = []
        all_selected_lig = []

        max_len = 0
        for each_dist in self.dist:  # (Nrec, Alig)
            rec_idx, lig_idx = torch.where(each_dist <= 8)
            rec_list = rec_idx.tolist()
            lig_list = lig_idx.tolist()

            rec_atom_indices_list.append(rec_list)
            lig_atom_indices_list.append(lig_list)

            all_selected_rec += rec_list
            all_selected_lig += lig_list

            if len(rec_list) > max_len:
                max_len = len(rec_list)

        all_selected_rec = sorted(set(all_selected_rec))
        all_selected_lig = sorted(set(all_selected_lig))

        # 수용체 XS 타입 업데이트(있으면)
        update_rec_xs = getattr(self.receptor, "update_rec_xs", None)
        if callable(update_rec_xs):
            for i in all_selected_rec:
                i_int = int(i)
                series = self.rec_index_to_series_dict.get(i_int, None)
                residue_index = (
                    self.heavy_atoms_residues_indices[i_int]
                    if i_int < len(self.heavy_atoms_residues_indices)
                    else None
                )
                update_rec_xs(self.rec_heavy_atoms_xs_types[i_int], i_int, series, residue_index)

        # 속성 마스크와 vdw 합 생성
        rec_is_hydro = {i: float(self.is_hydrophobic(i, is_lig=False)) for i in all_selected_rec}
        lig_is_hydro = {i: float(self.is_hydrophobic(i, is_lig=True)) for i in all_selected_lig}

        rec_is_donor = {i: float(self.is_hbdonor(i, is_lig=False)) for i in all_selected_rec}
        lig_is_donor = {i: float(self.is_hbdonor(i, is_lig=True)) for i in all_selected_lig}

        rec_is_accept = {i: float(self.is_hbacceptor(i, is_lig=False)) for i in all_selected_rec}
        lig_is_accept = {i: float(self.is_hbacceptor(i, is_lig=True)) for i in all_selected_lig}

        hydro_rows = []
        hbond_rows = []
        vdw_rows = []

        for rec_list, lig_list in zip(rec_atom_indices_list, lig_atom_indices_list):
            # 속성 벡터들(길이 K)
            r_hydro = torch.tensor([rec_is_hydro[i] for i in rec_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            l_hydro = torch.tensor([lig_is_hydro[i] for i in lig_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)

            r_donor = torch.tensor([rec_is_donor[i] for i in rec_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            l_donor = torch.tensor([lig_is_donor[i] for i in lig_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)

            r_accept = torch.tensor([rec_is_accept[i] for i in rec_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            l_accept = torch.tensor([lig_is_accept[i] for i in lig_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)

            # 패딩
            r_hydro = self._pad(r_hydro, max_len)
            l_hydro = self._pad(l_hydro, max_len)
            hydro_rows.append((r_hydro * l_hydro).unsqueeze(0))

            r_donor = self._pad(r_donor, max_len)
            l_donor = self._pad(l_donor, max_len)
            r_accept = self._pad(r_accept, max_len)
            l_accept = self._pad(l_accept, max_len)
            is_hbond = ((r_donor * l_accept + r_accept * l_donor) > 0).to(device_dtype_ref.dtype)
            hbond_rows.append(is_hbond.unsqueeze(0))

            # VdW 합
            vdw_sum = [
                self.vdw_radii_dict[self.rec_heavy_atoms_xs_types[r]] + self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[l]]
                for r, l in zip(rec_list, lig_list)
            ]
            vdw_vec = torch.tensor(vdw_sum, device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            vdw_rows.append(self._pad(vdw_vec, max_len).unsqueeze(0))

        self.rec_lig_is_hydrophobic = torch.cat(hydro_rows, dim=0) if hydro_rows else device_dtype_ref.new_zeros((self.number_of_poses, 0))
        self.rec_lig_is_hbond = torch.cat(hbond_rows, dim=0) if hbond_rows else device_dtype_ref.new_zeros((self.number_of_poses, 0))
        self.rec_lig_atom_vdw_sum = torch.cat(vdw_rows, dim=0) if vdw_rows else device_dtype_ref.new_zeros((self.number_of_poses, 0))

        # Vina용 거리 벡터(<=8Å, 0 제외) 추출 + 패딩
        vina_list = []
        for each_dist in self.dist:
            mask = (each_dist <= 8) & (each_dist > 0)
            vals = each_dist[mask]
            vina_list.append(self._pad(vals, max_len).unsqueeze(0))
        self.vina_dist = torch.cat(vina_list, dim=0)

        return self

    # -------------------------- 데이터 준비(분자 내부) --------------------------
    def _prepare_data_intra(self) -> "VinaSFTorch":
        device_dtype_ref = _like_of(self.intra_dist, self.pose_heavy_atoms_coords)

        rec_atom_indices_list = []
        lig_atom_indices_list = []
        all_selected_rec = []
        all_selected_lig = []

        max_len = 0
        for each_dist in self.intra_dist:  # (K,) 각 pose마다 pair 거리
            idx = torch.where(each_dist <= 8)[0].tolist()
            # 해당 idx는 self.ligand.intra_interacting_pairs의 인덱스
            rec_list = [self.ligand.intra_interacting_pairs[i][0] for i in idx]
            lig_list = [self.ligand.intra_interacting_pairs[i][1] for i in idx]

            rec_atom_indices_list.append(rec_list)
            lig_atom_indices_list.append(lig_list)
            all_selected_rec += rec_list
            all_selected_lig += lig_list

            if len(rec_list) > max_len:
                max_len = len(rec_list)

        all_selected_rec = sorted(set(all_selected_rec))
        all_selected_lig = sorted(set(all_selected_lig))

        # 내부 상호작용은 모두 리간드 원자들
        lig_is_hydro = {i: float(self.is_hydrophobic(i, is_lig=True)) for i in all_selected_lig + all_selected_rec}
        lig_is_donor = {i: float(self.is_hbdonor(i, is_lig=True)) for i in all_selected_lig + all_selected_rec}
        lig_is_accept = {i: float(self.is_hbacceptor(i, is_lig=True)) for i in all_selected_lig + all_selected_rec}

        hydro_rows = []
        hbond_rows = []
        vdw_rows = []

        for rec_list, lig_list in zip(rec_atom_indices_list, lig_atom_indices_list):
            r_hydro = torch.tensor([lig_is_hydro[i] for i in rec_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            l_hydro = torch.tensor([lig_is_hydro[i] for i in lig_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)

            r_donor = torch.tensor([lig_is_donor[i] for i in rec_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            l_donor = torch.tensor([lig_is_donor[i] for i in lig_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)

            r_accept = torch.tensor([lig_is_accept[i] for i in rec_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            l_accept = torch.tensor([lig_is_accept[i] for i in lig_list], device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)

            r_hydro = self._pad(r_hydro, max_len)
            l_hydro = self._pad(l_hydro, max_len)
            hydro_rows.append((r_hydro * l_hydro).unsqueeze(0))

            r_donor = self._pad(r_donor, max_len)
            l_donor = self._pad(l_donor, max_len)
            r_accept = self._pad(r_accept, max_len)
            l_accept = self._pad(l_accept, max_len)
            is_hbond = ((r_donor * l_accept + r_accept * l_donor) > 0).to(device_dtype_ref.dtype)
            hbond_rows.append(is_hbond.unsqueeze(0))

            vdw_sum = [
                self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[r]] + self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[l]]
                for r, l in zip(rec_list, lig_list)
            ]
            vdw_vec = torch.tensor(vdw_sum, device=device_dtype_ref.device, dtype=device_dtype_ref.dtype)
            vdw_rows.append(self._pad(vdw_vec, max_len).unsqueeze(0))

        self.intra_rec_lig_is_hydrophobic = torch.cat(hydro_rows, dim=0) if hydro_rows else device_dtype_ref.new_zeros((self.number_of_poses, 0))
        self.intra_rec_lig_is_hbond = torch.cat(hbond_rows, dim=0) if hbond_rows else device_dtype_ref.new_zeros((self.number_of_poses, 0))
        self.intra_rec_lig_atom_vdw_sum = torch.cat(vdw_rows, dim=0) if vdw_rows else device_dtype_ref.new_zeros((self.number_of_poses, 0))

        # intra용 거리 벡터
        vina_list = []
        for each_dist in self.intra_dist:
            vals = each_dist[each_dist > 0]  # 0 제외
            vina_list.append(self._pad(vals, max_len).unsqueeze(0))
        self.intra_vina_dist = torch.cat(vina_list, dim=0) if vina_list else device_dtype_ref.new_zeros((self.number_of_poses, 0))

        return self

    # -------------------------- 점수 계산 --------------------------
    def scoring(self) -> torch.Tensor:
        t1 = time.time()
        self.generate_pldist_mtrx()
        self._prepare_data()

        vina = VinaScoreCore(self.vina_dist, self.rec_lig_is_hydrophobic, self.rec_lig_is_hbond, self.rec_lig_atom_vdw_sum)
        try:
            vina_inter_term = vina.process()
            self.vina_inter_energy = vina_inter_term.reshape(-1, 1)
        except Exception:
            # 실패 시 큰 값으로 fallback
            like = _like_of(self.pose_heavy_atoms_coords)
            self.vina_inter_energy = torch.full((self.number_of_poses, 1), 99.99, device=like.device, dtype=like.dtype)

        vina_intra_term = torch.zeros_like(self.vina_inter_energy)
        if self.lig_intra_interacting_pairs:
            try:
                self.generate_intra_mtrx()
                self._prepare_data_intra()
                vina_intra = VinaScoreCore(
                    self.intra_vina_dist,
                    self.intra_rec_lig_is_hydrophobic,
                    self.intra_rec_lig_is_hbond,
                    self.intra_rec_lig_atom_vdw_sum,
                )
                vina_intra_term = vina_intra.process().reshape(-1, 1)
            except Exception:
                vina_intra_term = torch.zeros_like(self.vina_inter_energy)

        # 토션 보정
        torsion = 1 + 0.05846 * (self.ligand.active_torsion + 0.5 * self.ligand.inactive_torsion)
        return (self.vina_inter_energy + vina_intra_term) / torsion

    def score_and_gradient(self, ligand_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """입력 좌표(N, A, 3)에 대한 스코어와 그래디언트를 반환."""
        if self.ligand is None:
            raise ValueError("Ligand must be initialised before scoring.")

        coords = ligand_coords.clone().detach().requires_grad_(True)

        self.ligand.pose_heavy_atoms_coords = coords
        self.pose_heavy_atoms_coords = coords
        self.number_of_poses = coords.size(0)

        score = self.scoring()
        total = score.sum()
        total.backward()

        gradient = coords.grad.clone()
        detached = coords.detach()
        self.ligand.pose_heavy_atoms_coords = detached
        self.pose_heavy_atoms_coords = detached

        return score.detach(), gradient.detach().view_as(coords)

    # -------------------------- 원자 타입/속성 --------------------------
    def get_vdw_radii(self, xs: str) -> float:
        return self.vdw_radii_dict[xs]

    def get_vina_dist(self, r_index: int, l_index: int) -> torch.Tensor:
        return self.dist[:, r_index, l_index]

    def get_vina_rec_xs(self, index: int) -> str:
        return self.rec_heavy_atoms_xs_types[index]

    def get_vina_lig_xs(self, index: int) -> str:
        return self.updated_lig_heavy_atoms_xs_types[index]

    def is_hydrophobic(self, index: int, is_lig: bool) -> bool:
        atom_xs = (
            self.updated_lig_heavy_atoms_xs_types[index]
            if is_lig
            else self.rec_heavy_atoms_xs_types[index]
        )
        return atom_xs in ["C_H", "F_H", "Cl_H", "Br_H", "I_H"]

    def is_hbdonor(self, index: int, is_lig: bool = False) -> bool:
        atom_xs = (
            self.updated_lig_heavy_atoms_xs_types[index]
            if is_lig
            else self.rec_heavy_atoms_xs_types[index]
        )
        return atom_xs in ["N_D", "N_DA", "O_DA", "Met_D"]

    def is_hbacceptor(self, index: int, is_lig: bool = False) -> bool:
        atom_xs = (
            self.updated_lig_heavy_atoms_xs_types[index]
            if is_lig
            else self.rec_heavy_atoms_xs_types[index]
        )
        return atom_xs in ["N_A", "N_DA", "O_A", "O_DA"]

    # -------------------------- (선택) 반발 항 계산 유틸 --------------------------
    def cal_inter_repulsion(self, dist: torch.Tensor, vdw_sum: torch.Tensor) -> torch.Tensor:
        """vdW 합보다 짧은 거리에서의 상호 반발 항."""
        mask = (dist < vdw_sum).to(dist.dtype)  # (N, K)
        mask_sum = torch.sum(mask, dim=1)
        zero_idx = torch.where(mask_sum == 0)[0]
        if zero_idx.numel() > 0:
            # 0으로 떨어지는 분모 방지(의미 없는 큰 값으로 채움)
            mask[zero_idx, 0] = dist[zero_idx, 0].pow(20)

        term = torch.sum((mask * dist + (mask == 0).to(dist.dtype)).pow(-self.repulsive_), dim=1)
        term -= torch.sum(mask * dist, dim=1)
        self.inter_repulsive_term = term.reshape(-1, 1)
        return self.inter_repulsive_term


class VinaScoreCore:
    """Vina 에너지 항 계산의 코어(한 번 준비된 행렬에서 벡터화 계산)."""

    def __init__(self, dist_matrix: torch.Tensor, rec_lig_is_hydrophobic: torch.Tensor,
                 rec_lig_is_hbond: torch.Tensor, rec_lig_atom_vdw_sum: torch.Tensor) -> None:
        self.dist_matrix = dist_matrix
        self.rec_lig_is_hydro = rec_lig_is_hydrophobic
        self.rec_lig_is_hb = rec_lig_is_hbond
        self.rec_lig_atom_vdw_sum = rec_lig_atom_vdw_sum

    def score_function(self) -> torch.Tensor:
        d_ij = self.dist_matrix - self.rec_lig_atom_vdw_sum

        # 가우시안 항
        gauss1 = torch.sum(torch.exp(-((d_ij / 0.5) ** 2)), dim=1) - torch.sum((d_ij == 0).to(d_ij.dtype), dim=1)
        # exp(-9/4)는 상수(≈ e^-2.25)
        exp_const = math.exp(-9.0 / 4.0)
        gauss2 = torch.sum(torch.exp(-(((d_ij - 3.0) / 2.0) ** 2)), dim=1) - torch.sum(
            (d_ij == 0).to(d_ij.dtype) * exp_const, dim=1
        )

        # 반발(거리 부족)
        repulsion = torch.sum(torch.where(d_ij < 0, d_ij, d_ij.new_zeros(1)).pow(2), dim=1)

        # 소수성
        hydro1 = self.rec_lig_is_hydro * (d_ij <= 0.5).to(d_ij.dtype)
        hydro2_mask = self.rec_lig_is_hydro * ((d_ij > 0.5) & (d_ij < 1.5)).to(d_ij.dtype)
        hydro2 = 1.5 * hydro2_mask - hydro2_mask * d_ij
        hydrophobic = torch.sum(hydro1 + hydro2, dim=1)

        # H-결합
        hb1 = self.rec_lig_is_hb * (d_ij <= -0.7).to(d_ij.dtype)
        hb2 = self.rec_lig_is_hb * ((d_ij < 0) & (d_ij > -0.7)).to(d_ij.dtype) * (-d_ij) / 0.7
        hbond = torch.sum(hb1 + hb2, dim=1)

        # 가중 합
        inter_energy = (
            -0.035579 * gauss1
            - 0.005156 * gauss2
            + 0.840245 * repulsion
            - 0.035069 * hydrophobic
            - 0.587439 * hbond
        )
        return inter_energy

    def process(self) -> torch.Tensor:
        return self.score_function()
