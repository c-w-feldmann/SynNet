"""Draw molecules as images."""

import uuid
import warnings
from pathlib import Path
from typing import Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from rdkit import Chem
from rdkit.Chem import Draw

from synnet.utils.custom_types import PathType


class MolDrawer:
    """Draws molecules as images."""

    outfolder: Path
    lookup: dict[str, str]

    def __init__(self, path: PathType, subfolder: PathType = "assets"):
        """Initialize the drawer.

        Parameters
        ----------
        path : PathType
            Path to the output folder.
        subfolder : str, optional
            Subfolder to store the images, by default "assets".
        """
        # Init outfolder
        if path and not Path(path).is_dir():
            raise NotADirectoryError(path)
        self.outfolder = Path(path) / subfolder
        self.outfolder.mkdir(exist_ok=True)

        # Placeholder
        self.lookup: dict[str, str] = {}

    def _hash(self, smiles: list[str]) -> Self:
        """Hash the smiles to get a unique filename

        Goal: Get a short, valid, and hopefully unique filename for each molecule.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.

        Returns
        -------
        Self
            The current instance.
        """
        self.lookup = {smile: str(uuid.uuid4())[:8] for smile in smiles}
        return self

    def get_path(self) -> Path:
        """Get the output folder.

        Returns
        -------
        Path
            The output folder.
        """
        return self.outfolder

    def get_molecule_filesnames(self) -> dict[str, str]:
        """Get the lookup table of SMILES and filenames.

        Returns
        -------
        dict[str, str]
            The lookup table of SMILES and filenames.
        """
        return self.lookup

    def plot(self, smiles_list: Union[list[str], str]) -> Self:
        """Plot smiles as 2d molecules and save to `self.path/subfolder/*.svg`.

        Parameters
        ----------
        smiles_list : Union[list[str], str]
            List of SMILES strings.

        Returns
        -------
        Self
            The current instance.
        """
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

            with open(fname, "w", encoding="UTF-8") as f:
                f.write(p)

        return self
