"""Computes the fingerprint similarity of molecules in {valid,test}-set to molecules in the training set."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from synnet.config import MAX_PROCESSES
from synnet.utils.data_utils import SyntheticTreeSet


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory with `*{train,valid,test}*.json.gz`-data of synthetic trees",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save similarity-values for test,valid-synthetic trees. (*csv.gz)",
    )
    # Processing
    parser.add_argument(
        "--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus"
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def _match_dataset_filename(path: str, dataset_type: str) -> Path:
    """Helper to find the exact filename for {train,valid,test} file."""
    files = list(Path(path).glob(f"*{dataset_type}*.json.gz"))
    if len(files) != 1:
        raise ValueError(f"Can not find unique '{dataset_type} 'file, got {files}")
    return files[0]


def find_similar_fp(
    fp: npt.ArrayLike, fps_reference: npt.ArrayLike
) -> tuple[float, np.int_]:
    """Finds most similar fingerprint in a reference set for `fp`.
    Uses Tanimoto Similarity.
    """
    dists = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps_reference))
    similarity_score = float(dists.max())
    idx = dists.argmax()
    return similarity_score, idx


def _compute_fp_bitvector(
    smiles: list[str], radius: int = 2, nbits: int = 1024
) -> list[npt.NDArray[Any]]:
    return [
        np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smi), radius, nBits=nbits
            )
        )
        for smi in smiles
    ]


def get_smiles_and_fps(dataset: str) -> tuple[list[str], npt.NDArray[Any]]:
    file = _match_dataset_filename(args.input_dir, dataset)
    syntree_collection = SyntheticTreeSet().load(file)
    smiles_list = []
    for syntenic_tree in syntree_collection.synthetic_tree_list:
        if syntenic_tree.root is not None:
            if syntenic_tree.root.smiles is not None:
                smiles_list.append(syntenic_tree.root.smiles)
    fps = np.vstack(_compute_fp_bitvector(smiles_list))
    return smiles_list, fps


def compute_most_similar_smiles(
    split: str,
    fps: npt.NDArray[Any],
    smiles: list[str],
    /,
    fps_reference: npt.NDArray[Any],
    smiles_reference: list[str],
) -> pd.DataFrame:
    func = partial(find_similar_fp, fps_reference=fps_reference)
    with mp.Pool(processes=args.ncpu) as pool:
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


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")
    # Load data
    smiles_train, fps_train = get_smiles_and_fps("train")
    smiles_valid, fps_valid = get_smiles_and_fps("valid")
    smiles_test, fps_test = get_smiles_and_fps("test")

    # Compute (mp)
    logger.info("Start computing most similar smiles...")
    df_valid = compute_most_similar_smiles(
        "valid",
        fps_valid,
        smiles_valid,
        fps_reference=fps_train,
        smiles_reference=smiles_train,
    )
    df_test = compute_most_similar_smiles(
        "test",
        fps_test,
        smiles_test,
        fps_reference=fps_train,
        smiles_reference=smiles_train,
    )
    logger.info("Computed most similar smiles for {valid,test}-set.")

    # Save
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([df_valid, df_test], axis=0, ignore_index=True)
    df.to_csv(args.output_file, index=False, compression="gzip")
    logger.info(f"Successfully saved output to {args.output_file}.")

    logger.info("Completed.")
