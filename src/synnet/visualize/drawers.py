import uuid
import warnings
from pathlib import Path
from typing import Optional, Union
try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import rdkit.Chem as Chem
from rdkit.Chem import Draw


class MolDrawer:
    """Draws molecules as images."""

    outfolder: Path
    lookup: dict[str, str]

    def __init__(self, path: Optional[str], subfolder: str = "assets"):
        # Init outfolder
        if not (path is not None and Path(path).exists()):
            raise NotADirectoryError(path)
        self.outfolder = Path(path) / subfolder
        self.outfolder.mkdir(exist_ok=True)

        # Placeholder
        self.lookup: dict[str, str] = {}

    def _hash(self, smiles: list[str]) -> Self:
        """Hashing for amateurs.
        Goal: Get a short, valid, and hopefully unique filename for each molecule."""
        self.lookup = {smile: str(uuid.uuid4())[:8] for smile in smiles}
        return self

    def get_path(self) -> Path:
        return self.outfolder

    def get_molecule_filesnames(self) -> dict[str, str]:
        return self.lookup

    def plot(self, smiles_list: Union[list[str], str]) -> Self:
        """Plot smiles as 2d molecules and save to `self.path/subfolder/*.svg`."""
        if isinstance(smiles_list, str):
            warnings.warn("Please use a list of smiles instead of a single string.")
            smiles_list = [smiles_list]
        self._hash(smiles_list)

        for k, v in self.lookup.items():
            fname = self.outfolder / f"{v}.svg"
            mol = Chem.MolFromSmiles(k)
            # Plot
            drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 150)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            p = drawer.GetDrawingText()

            with open(fname, "w") as f:
                f.write(p)

        return self


if __name__ == "__main__":
    pass
