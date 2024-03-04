import numba
import numpy as np
import numpy.typing as npt

from synnet.encoding.embedding import MorganFingerprintEmbedding


@numba.njit()
def cosine_distance(v1: npt.NDArray[np.float_], v2: npt.NDArray[np.float_]) -> float:
    """Compute the cosine distance between two 1d-vectors.

    Note:
        cosine_similarity = x'y / (||x|| ||y||) in [-1,1]
        cosine_distance   = 1 - cosine_similarity in [0,2]
    """
    return max(
        0, min(1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 2)
    )


def ce_distance(
    y: npt.NDArray[np.float_],
    y_pred: npt.NDArray[np.float_],
    eps: float = 1e-15,
) -> float:
    """Computes the cross-entropy between two vectors.

    Args:
        y (np.ndarray): First vector.
        y_pred (np.ndarray): Second vector.
        eps (float, optional): Small value, for numerical stability. Defaults
            to 1e-15.

    Returns:
        float: The cross-entropy.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))


@numba.njit()
def _tanimoto_similarity(
    fp1: npt.NDArray[np.float_], fp2: npt.NDArray[np.float_]
) -> float:
    """
    Returns the Tanimoto similarity between two molecular fingerprints.

    Args:
        fp1 (np.ndarray): Molecular fingerprint 1.
        fp2 (np.ndarray): Molecular fingerprint 2.

    Returns:
        float: Tanimoto similarity.
    """
    return np.sum(fp1 * fp2) / (np.sum(fp1) + np.sum(fp2) - np.sum(fp1 * fp2))


def tanimoto_similarity(
    target_fp: npt.NDArray[np.float_], smis: list[str]
) -> npt.NDArray[np.float_]:
    """
    Returns the Tanimoto similarities between a target fingerprint and molecules
    in an input list of SMILES.

    Parameters
    ----------
    target_fp: np.ndarray
        Contains the reference (target) fingerprint.
    smis: list[str]
        Contains SMILES to compute similarity to.

    Returns
    -------
    npt.NDArray[np.float_]
        Tanimoto similarities.
    """
    morgan_fp = MorganFingerprintEmbedding(radius=2, n_bits=4096)
    fps = [morgan_fp.transform_smiles(smi) for smi in smis]
    return np.array([_tanimoto_similarity(target_fp, fp) for fp in fps])
