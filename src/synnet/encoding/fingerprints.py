from typing import Optional

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import AllChem


def fp_embedding(
    smi: Optional[str], _radius: int = 2, _nBits: int = 4096
) -> npt.NDArray[np.float_]:
    """
    General function for building variable-size & -radius Morgan fingerprints.

    Args:
        smi (str): The SMILES to encode.
        _radius (int, optional): Morgan fingerprint radius. Defaults to 2.
        _nBits (int, optional): Morgan fingerprint length. Defaults to 4096.

    Returns:
        np.ndarray: A Morgan fingerprint generated using the specified parameters.
    """
    if smi is None:
        return np.zeros(_nBits, dtype=np.float_).reshape((-1,))
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits), dtype=np.float_
        )
        return features_vec.reshape((-1,))
