"""Distance functions for comparing molecular fingerprints."""

import numba
import numpy as np
import numpy.typing as npt

from synnet.encoding.embedding import MorganFingerprintEmbedding


@numba.njit()
def cosine_distance(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]) -> float:
    """Compute the cosine distance between two 1d-vectors.

    Notes
    -----
    cosine_similarity = x'y / (||x|| ||y||) in [-1,1]
    cosine_distance   = 1 - cosine_similarity in [0,2]

    Parameters
    ----------
    v1 : npt.NDArray[np.float64]
        First vector.
    v2 : npt.NDArray[np.float64]
        Second vector.

    Returns
    -------
    float
        The cosine distance
    """
    return max(
        0, min(1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 2)
    )


def ce_distance(
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    eps: float = 1e-15,
) -> float:
    """Compute the cross-entropy between two vectors.

    Parameters
    ----------
    y : npt.NDArray[np.float64]
        First vector.
    y_pred : npt.NDArray[np.float64]
        Second vector.
    eps : float, optional
        Small value, for numerical stability, by default 1e-15

    Returns
    -------
    float
        The cross-entropy.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))


@numba.njit()
def _tanimoto_similarity(
    fp1: npt.NDArray[np.int_], fp2: npt.NDArray[np.float64]
) -> float:
    """Return the Tanimoto similarity between two molecular fingerprints.

    Parameters
    ----------
    fp1 : npt.NDArray[np.int_]
        Molecular fingerprint 1.
    fp2 : npt.NDArray[np.float64]
        Molecular fingerprint 2.

    Returns
    -------
    float
        Tanimoto similarity.
    """
    return np.sum(fp1 * fp2) / (np.sum(fp1) + np.sum(fp2) - np.sum(fp1 * fp2))


def tanimoto_similarity(
    target_fp: npt.NDArray[np.int_], smis: list[str]
) -> npt.NDArray[np.float64]:
    """Calculate Tanimoto similarities.

    Tanimoto similarities between a target fingerprint and molecules
    in an input list of SMILES.

    Parameters
    ----------
    target_fp: np.ndarray
        Contains the reference (target) fingerprint.
    smis: list[str]
        Contains SMILES to compute similarity to.

    Returns
    -------
    npt.NDArray[np.float64]
        Tanimoto similarities.
    """
    morgan_fp = MorganFingerprintEmbedding(radius=2, n_bits=4096)
    fps = [morgan_fp.transform_smiles(smi) for smi in smis]
    return np.array([_tanimoto_similarity(target_fp, fp) for fp in fps])
