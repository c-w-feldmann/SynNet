"""Computes the fingerprint similarity of molecules in {valid,test}-set to molecules in the training set."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from rdkit import DataStructs

from synnet.config import MAX_PROCESSES
from synnet.encoding.embedding import MorganFingerprintEmbedding
from synnet.utils.custom_types import PathType
from synnet.utils.data_utils import SyntheticTreeSet


def _match_dataset_filename(path: PathType, dataset_type: str) -> Path:
    """Find the exact filename for {train,valid,test} file.

    Parameters
    ----------
    path: PathType
        The path to search for the dataset file.
    dataset_type: str
        The dataset type to search for.

    Returns
    -------
    Path
        The path to the dataset file.
    """
    files = list(Path(path).glob(f"*{dataset_type}*.json.gz"))
    if len(files) != 1:
        raise ValueError(f"Can not find unique '{dataset_type} 'file, got {files}")
    return files[0]


def find_similar_fp(
    fp: npt.ArrayLike, fps_reference: npt.ArrayLike
) -> tuple[float, int]:
    """Find fingerprint with highest Tanimoto similarity to `fp` in `fps_reference`.

    Parameters
    ----------
    fp : npt.ArrayLike
        The fingerprint to compare.
    fps_reference: npt.ArrayLike
        The reference fingerprints to compare against.

    Returns
    -------
    float
        The similarity score.
    idx
        The index of the most similar fingerprint.
    """
    dists = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps_reference))
    similarity_score = float(dists.max())
    idx = dists.argmax()
    return similarity_score, int(idx)


def get_smiles_and_fps(
    input_dir: PathType, dataset: str
) -> tuple[list[str], npt.NDArray[Any]]:
    """Load SMILES and fingerprints from dataset.

    Parameters
    ----------
    input_dir : PathType
        The input directory.
    dataset : str
        The dataset to load.

    Returns
    -------
    list[str]
        The SMILES strings.
    npt.NDArray[Any]
        The fingerprint matrix.
    """
    file = _match_dataset_filename(input_dir, dataset)
    syntree_collection = SyntheticTreeSet().load(file)
    smiles_list = []
    for syntenic_tree in syntree_collection.synthetic_tree_list:
        if syntenic_tree.root is not None:
            if syntenic_tree.root.smiles is not None:
                smiles_list.append(syntenic_tree.root.smiles)
    morgan_embedding = MorganFingerprintEmbedding(radius=2, n_bits=1024)
    fps = np.vstack([morgan_embedding.transform_smiles(smi) for smi in smiles_list])
    return smiles_list, fps


def compute_most_similar_smiles(
    split: str,
    fps: npt.NDArray[Any],
    smiles: list[str],
    fps_reference: npt.NDArray[Any],
    smiles_reference: list[str],
    ncpu: int = MAX_PROCESSES,
) -> pd.DataFrame:
    """Compute the most similar smiles in `smiles_reference` to `smiles`.

    Parameters
    ----------
    split : str
        The dataset split.
    fps : npt.NDArray[Any]
        The fingerprints to compare.
    smiles : list[str]
        The SMILES strings.
    fps_reference : npt.NDArray[Any]
        The reference fingerprints.
    smiles_reference : list[str]
        The reference SMILES strings.
    ncpu : int, optional
        The number of cpus to use, by default MAX_PROCESSES

    Returns
    -------
    pd.DataFrame
        The DataFrame with the most similar smiles.
    """
    func = partial(find_similar_fp, fps_reference=fps_reference)
    with mp.Pool(processes=ncpu) as pool:
        results = pool.map(func, fps)

    similarities, idx = zip(*results)
    most_similiar_ref_smiles = np.asarray(smiles_reference)[np.asarray(idx, dtype=int)]
    # ^ Use numpy for slicing...

    df = pd.DataFrame(
        {
            "split": split,
            "smiles": smiles,
            "most_similar_smiles": most_similiar_ref_smiles,
            "similarity": similarities,
        }
    )
    return df


@click.command(name="identify-similar-fps")
@click.argument(
    "input_dir",
    type=click.Path(exists=True),
    help="Directory with `*{train,valid,test}*.json.gz`-data of synthetic trees",
)
@click.argument(
    "output_file",
    type=click.Path(),
    help="File to save similarity-values for test,valid-synthetic trees. (*csv.gz)",
)
@click.option(
    "--ncpu",
    type=int,
    default=MAX_PROCESSES,
    help="Number of cpus",
)
def identify_similar_fps(input_file: str, output_file: str, ncpu: int) -> None:
    """Identify similar fingerprints.

    Parameters
    ----------
    input_file : str
        The input file.
    output_file : str
        The output file.
    ncpu : int
        The number of cpus.
    """
    logger.info(f"Loading data from {input_file}")
    # Load data
    smiles_train, fps_train = get_smiles_and_fps(input_file, "train")
    smiles_valid, fps_valid = get_smiles_and_fps(input_file, "valid")
    smiles_test, fps_test = get_smiles_and_fps(input_file, "test")

    # Compute (mp)
    logger.info("Start computing most similar smiles...")
    df_valid = compute_most_similar_smiles(
        "valid",
        fps_valid,
        smiles_valid,
        fps_reference=fps_train,
        smiles_reference=smiles_train,
        ncpu=ncpu,
    )
    df_test = compute_most_similar_smiles(
        "test",
        fps_test,
        smiles_test,
        fps_reference=fps_train,
        smiles_reference=smiles_train,
        ncpu=ncpu,
    )
    logger.info("Computed most similar smiles for {valid,test}-set.")

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([df_valid, df_test], axis=0, ignore_index=True)
    df.to_csv(output_file, index=False, compression="gzip")
    logger.info(f"Successfully saved output to {output_file}.")

    logger.info("Completed.")


if __name__ == "__main__":
    identify_similar_fps()
