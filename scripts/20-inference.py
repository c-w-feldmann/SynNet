"""Script to decode molecules from a file."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from time import time
from typing import Any, Optional, Union

import click
import pandas as pd
from loguru import logger
from rdkit import RDLogger

from synnet.config import MAX_PROCESSES
from synnet.data_generation.syntrees import MorganFingerprintEncoder
from synnet.decoding.decoder import (
    HelperDataloader,
    SynTreeDecoder,
    SynTreeDecoderGreedy,
)
from synnet.encoding.distances import tanimoto_similarity
from synnet.encoding.embedding import MolecularEmbeddingManager
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.utils.custom_types import PathType
from synnet.utils.data_utils import ReactionSet, SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel
from synnet.utils.synnet_exceptions import FailedReconstructionError


def wrapper(
    target: str,
    *,
    syntree_decoder: Union[SynTreeDecoder, SynTreeDecoderGreedy],
    mol_encoder: MorganFingerprintEncoder,
    **kwargs: Any,
) -> tuple[SyntheticTree, Optional[float]]:
    """Wrapper function to decode targets into `SyntheticTree` & catch Exceptions.

    Notes
    -----
        Always return a Dict *with* a `SyntheticTree`.
        This allows easily keep everything in order when saving to a file via `SyntheticTreeSet` and
        allows to call its method `is_valid()` on the entire list.

    Parameters
    ----------
    target : str
        The target SMILES string.
    syntree_decoder : Union[SynTreeDecoder, SynTreeDecoderGreedy]
        The decoder.
    mol_encoder : MorganFingerprintEncoder
        The encoder.
    **kwargs : Any
        Additional keyword arguments for the decoder.
    """
    try:
        z_target = mol_encoder.encode(target)
    except ValueError as e:
        logger.error(f"Failed to encode {target}: {e}")
        return SyntheticTree(), None

    try:
        res = syntree_decoder.decode(z_target, **kwargs)
    except FailedReconstructionError as e:
        logger.error(f"Failed to encode {target}: {e}")
        return SyntheticTree(), None

    return res


def print_stats(df: pd.DataFrame, data: str) -> None:
    """Log some statistics about the results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the results.
    data : str
        File with molecules to decode.
    """
    n_valid = df["is_valid"].sum()
    n_recovered = (df["max_similarity"] == 1.0).sum()
    recovery_rate = n_recovered / n_valid
    avg_similarity = df["max_similarity"][df["is_valid"]].mean()

    logger.info(f"For {data}:")
    logger.info(f"  Total number of targets: {len(df)}")
    logger.info(f"  Total number of valid reconstructions: {n_valid}")
    logger.info(f"  Total number of successful reconstructions: {n_recovered}")
    logger.info(f"  {recovery_rate=}  ({recovery_rate :.4%})")
    logger.info(f"  {avg_similarity=} ({avg_similarity:.4%})")
    return None


def postprocess_results(
    tree_list: list[SyntheticTree],
    similarity_array: Optional[list[Optional[float]]] = None,
) -> pd.DataFrame:
    """Postprocess the results.

    Parameters
    ----------
    tree_list : list[SyntheticTree]
        List of decoded syntrees.
    similarity_array : Optional[list[Optional[float]]], optional
        List of similarities, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with the results.
    """
    df = pd.DataFrame()
    df["syntree"] = tree_list
    if similarity_array is not None:
        df["similarity"] = similarity_array

    df["is_valid"] = df["syntree"].apply(lambda x: x.is_valid)

    df["decoded_smiles"] = df["syntree"].apply(
        lambda st: st.root.smiles if st.is_valid else None
    )
    df["decoded_depth"] = df["syntree"].apply(
        lambda st: st.depth if st.is_valid else None
    )

    return df


def save_results(output_dir: PathType, df: pd.DataFrame) -> None:
    """Save the results.

    Parameters
    ----------
    output_dir : PathType
        Directory to save output.
    df : pd.DataFrame
        Dataframe with the results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save df
    syntrees = df.pop("syntree").to_list()
    df.to_csv(output_dir / "results.csv", index=False)

    # Save generated trees
    syntree_collection = SyntheticTreeSet(sts=syntrees)
    syntree_collection.save(output_dir / "decoded_syntrees.json.gz")
    logger.info(f"Saved results to {output_dir}.")


def _setup_loggers(verbose: bool, debug: bool, output_dir: PathType) -> None:
    """Set up the loggers.

    Parameters
    ----------
    verbose: bool
        Make logger verbose.
    debug: bool
        Enable debug mode for logger.
    output_dir:
        Path to save logs.
    """
    if verbose:
        log_path = Path(output_dir) / "inference.log"
        log_path.parent.mkdir(exist_ok=True, parents=True)
        logger.add(log_path, level="INFO")

    if debug:
        logger.level("DEBUG")
    else:
        RDLogger.DisableLog("rdApp.*")


@click.command()
@click.argument(
    "rxns_collection_file",
    type=click.Path(exists=True),
    help="Input file for the collection of reactions matched with building-blocks.",
)
@click.argument(
    "embeddings_knn_file",
    type=click.Path(exists=True),
    help="Input file for the pre-computed embeddings (*.npy).",
)
@click.argument(
    "ckpt_dir",
    type=click.Path(exists=True),
    help="Directory with checkpoints for {act,rt1,rxn,rt2}-model.",
)
@click.argument(
    "output_dir",
    type=click.Path(),
    help="Directory to save output.",
)
@click.option(
    "--num",
    type=int,
    default=-1,
    help="Number of molecules to predict.",
)
@click.option(
    "--data",
    type=str,
    help="File with molecules to decode.",
)
@click.option(
    "--ncpu",
    type=int,
    default=MAX_PROCESSES,
    help="Number of cpus",
)
@click.option(
    "--verbose",
    default=False,
    is_flag=True,
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
)
def inference(
    rxns_collection_file: str,
    embeddings_knn_file: str,
    ckpt_dir: PathType,
    output_dir: str,
    num: int,
    data: str,
    ncpu: int,
    verbose: bool,
    debug: bool,
) -> None:
    """Decode molecules from a file.

    Parameters
    ----------
    rxns_collection_file : str
        Input file for the collection of reactions matched with building-blocks.
    embeddings_knn_file : str
        Input file for the pre-computed embeddings (*.npy).
    ckpt_dir : str
        Directory with checkpoints for {act,rt1,rxn,rt2}-model.
    output_dir : str
        Directory to save output.
    num : int
        Number of molecules to predict.
    data : str
        File with molecules to decode.
    ncpu : int
        Number of cpus.
    verbose : bool
        Make logger verbose.
    debug : bool
        Enable debug mode for logger.
    """
    logger.info("Start.")
    t0 = time()

    _setup_loggers(verbose, debug, output_dir)

    # region-dataloading
    # Load molecules to decode
    targets_all = HelperDataloader().fetch_data(data)
    if num > 0:  # Select only n queries
        targets = targets_all[:num]
    else:
        targets = targets_all
    logger.info(f"Number of targets, i.e. mols to decode, : {len(targets)}")

    # Load assets
    reaction_collection = ReactionSet().load(rxns_collection_file)

    # Load and init building blocks embedder (kdtree)
    bblocks_molembedder = MolecularEmbeddingManager.from_folder(embeddings_knn_file)

    # Load models
    logger.info("Start loading models from checkpoints...")

    ckpt_files = [
        find_best_model_ckpt(Path(ckpt_dir) / model)
        for model in ["act", "rt1", "rxn", "rt2"]
    ]
    act_net, rt1_net, rxn_net, rt2_net = [
        load_mlp_from_ckpt(file) for file in ckpt_files
    ]

    logger.info("...loading models completed.")
    # endregion-dataloading
    # Simple Encoder
    stdecoder = SynTreeDecoder(
        building_blocks_embedding_manager=bblocks_molembedder,
        reaction_collection=reaction_collection,
        action_net=act_net,
        reactant1_net=rt1_net,
        rxn_net=rxn_net,
        reactant2_net=rt2_net,
        similarity_fct=tanimoto_similarity,
    )
    # Greedy decoder
    stdecoder_greedy = SynTreeDecoderGreedy(decoder=stdecoder)

    # Decode targets
    _wrapper = partial(
        wrapper,
        syntree_decoder=stdecoder_greedy,
        mol_encoder=MorganFingerprintEncoder(2, 4096),
    )

    logger.info(f"Start decoding {len(targets)} targets.")

    result_list = chunked_parallel(targets, _wrapper, max_cpu=ncpu, verbose=verbose)
    syn_tree_list: list[SyntheticTree] = []
    similarity_list: list[Optional[float]] = []
    for syn_tree, similarity in result_list:
        syn_tree_list.append(syn_tree)
        similarity_list.append(similarity)
    logger.info("Completed decoding.")
    logger.info(f"Elapsed: {time()-t0:.0f}s")

    # Convert results to df
    df = postprocess_results(syn_tree_list, similarity_list)

    # Print some stats
    print_stats(df, data)

    # Save results
    save_results(output_dir, df)
    logger.info("Completed.")


if __name__ == "__main__":
    inference()
