"""Compute the mean reciprocal ranking for reactant 1.

selection using the different distance metrics in the k-NN search.
"""
from typing import Any, Callable

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

import click
import numpy as np
from loguru import logger
from tqdm import tqdm

from synnet.config import MAX_PROCESSES
from synnet.encoding.distances import ce_distance, cosine_distance
from synnet.encoding.embedding import MolecularEmbeddingManager
from synnet.models.common import load_mlp_from_ckpt, xy_to_dataloader
from synnet.utils.custom_types import PathType


@click.command(name="compute-mrr")
@click.argument(
    "ckpt_file", type=click.Path(exists=True, dir_okay=False, file_okay=False)
)
@click.argument(
    "embeddings_file", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "feature_matrix_file", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument(
    "target_matrix_file", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option(
    "--ncpu",
    type=int,
    default=MAX_PROCESSES,
    help="Number of cpus",
)
@click.option(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    help="Device to use for computation",
)
@click.option(
    "--distance",
    type=str,
    default="euclidean",
    help="Distance function for `BallTree`.",
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Flag to run in debug mode",
)
def compute_mean_reciprocal_ranking(
    ckpt_file: PathType,
    embeddings_file: PathType,
    feature_matrix_file: PathType,
    target_matrix_file: PathType,
    ncpu: int = MAX_PROCESSES,
    batch_size: int = 64,
    device: str = "cuda:0",
    distance: str = "euclidean",
    debug: bool = False,
) -> None:
    """Compute the mean reciprocal ranking for reactant 1 selection using the different distance metrics in the k-NN search.

    Parameters
    ----------
    ckpt_file : PathType
        Path to the checkpoint file.
    embeddings_file : PathType
        Path to the embeddings file.
    feature_matrix_file : PathType
        Path to the feature matrix file.
    target_matrix_file : PathType
        Path to the target matrix file.
    ncpu : int, optional
        Number of cpus, by default MAX_PROCESSES
    batch_size : int, optional
        Batch size, by default 64
    device : str, optional
        Device to use for computation, by default "cuda:0"
    distance : str, optional
        Distance function for `BallTree`, by default "euclidean"
    debug : bool, optional
        Flag to run in debug mode, by default False
    """
    logger.info("Start.")

    # Init BallTree for kNN-search
    metric: str | Callable[[Any], float]
    if distance == "cross_entropy":
        metric = ce_distance
    elif distance == "cosine":
        metric = cosine_distance
    else:
        metric = distance

    # Recall default: Morgan fingerprint with radius=2, nbits=256
    mol_embedder = MolecularEmbeddingManager.from_folder(embeddings_file)
    mol_embedder.init_balltree(metric=metric)
    n, d = mol_embedder.embeddings.shape

    # Load data
    dataloader = xy_to_dataloader(
        X_file=feature_matrix_file,
        y_file=target_matrix_file,
        task="classification",
        n=None if not debug else 128,
        batch_size=batch_size,
        num_workers=ncpu,
        shuffle=False,
    )

    # Load MLP
    rt1_net = load_mlp_from_ckpt(ckpt_file)
    rt1_net.to(device)

    rank_list = []
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        y_hat = rt1_net(X)  # (batch_size,nbits)

        ind_true = mol_embedder.kdtree.query(
            y.detach().cpu().numpy(), k=1, return_distance=False
        )
        ind = mol_embedder.kdtree.query(
            y_hat.detach().cpu().numpy(), k=n, return_distance=False
        )

        irows, icols = np.nonzero(
            ind == ind_true
        )  # irows = range(batch_size), icols = ranks
        rank_list.append(icols)

    ranks_array = np.asarray(rank_list, dtype=int).flatten()  # (nSamples,)
    rrs = 1 / (ranks_array + 1)  # +1 for offset 0-based indexing

    logger.info(f"Result using metric: {metric}")
    logger.info(f"The mean reciprocal ranking is: {rrs.mean():.3f}")
    TOP_N_RANKS = (1, 3, 5, 10, 15, 30)
    for i in TOP_N_RANKS:
        n_recovered = sum(ranks_array < i)
        n = len(ranks_array)
        logger.info(
            f"The Top-{i:<2d} recovery rate is: {n_recovered/n:.3f} ({n_recovered}/{n})"
        )


if __name__ == "__main__":
    compute_mean_reciprocal_ranking()
