# VinaSF PyTorch

This repository provides a PyTorch-friendly version of the Vina scoring function and the
OpenDock sampling framework. It is a fork of the excellent
[guyuehuo/opendock](https://github.com/guyuehuo/opendock) project.

## Installation

### Install from Git (recommended)
You can install the latest version directly from GitHub with `pip`:

```bash
pip install git+https://github.com/kim-iljung/VinaSF_PyTorch.git
```

This command uses the `pyproject.toml` configuration in the repository so no additional
flags (such as `#subdirectory=`) are needed.

### Local development install
If you have cloned the repository locally, install it in editable mode to pick up changes
without reinstalling:

```bash
pip install -e .
```

## Usage example
The snippet below shows how to evaluate the Vina scoring function on a receptor/ligand pair
using RDKit and PyTorch. The same pattern works for coordinates you generate elsewhere.

```python
from rdkit import Chem
import torch

from vinasf_torch import VinaSFTorch

receptor = Chem.MolFromPDBFile("receptor.pdb", removeHs=False)
ligand = Chem.SDMolSupplier("ligand.sdf", removeHs=False)[0]

vina = VinaSFTorch.from_rdkit(receptor, ligand).to("cuda")

# Use either the current ligand coordinates or supply your own tensor with
# shape (N, A, 3) for N poses and A heavy atoms.
coords = vina.ligand.pose_heavy_atoms_coords.clone()
score, gradient = vina.score_and_gradient(coords)

print("Score per pose:", score)
print("Gradient tensor:", gradient)
```

Calling ``.to(device)`` on ``VinaSFTorch`` moves both the module and any cached
receptor/ligand tensors to the requested device, so the entire score evaluation
can run on GPU without manual tensor transfers.

## Contributing
Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a
pull request so that improvements can be discussed.

## License
This project is distributed under the terms of the [MIT License](LICENSE).
