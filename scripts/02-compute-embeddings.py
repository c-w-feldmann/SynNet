"""
Computes the molecular embeddings of the purchasable building blocks.

The embeddings are also referred to as "output embedding".
In the embedding space, a kNN-search will identify the 1st or 2nd reactant.
"""

import argparse
import json
import logging

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.embedding import (
    MolecularEmbeddingManager,
    MorganFingerprintEmbedding,
)

logger = logging.getLogger(__file__)

FUNCTIONS: dict[str, MorganFingerprintEmbedding] = {
    "fp_4096": MorganFingerprintEmbedding(radius=2, n_bits=4096),
    "fp_2048": MorganFingerprintEmbedding(radius=2, n_bits=2048),
    "fp_1024": MorganFingerprintEmbedding(radius=2, n_bits=1024),
    "fp_512": MorganFingerprintEmbedding(radius=2, n_bits=512),
    "fp_256": MorganFingerprintEmbedding(radius=2, n_bits=256),
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Output file for precomputed embeddings.",
    )
    parser.add_argument(
        "--featurization-fct",
        type=str,
        choices=FUNCTIONS.keys(),
        help="Featurization function applied to each molecule.",
    )
    # Processing
    parser.add_argument(
        "--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus"
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    logger.info(f"Successfully read {args.building_blocks_file}.")
    logger.info(f"Total number of building blocks: {len(bblocks)}.")

    # Compute embeddings
    embedding_method: MorganFingerprintEmbedding = FUNCTIONS[args.featurization_fct]

    mol_embedder = MolecularEmbeddingManager(
        smiles_list=bblocks,
        embedding_method=embedding_method,
        n_jobs=args.ncpu,
    )

    # Save?
    mol_embedder.to_folder(args.output_folder)
    logger.info("Completed.")
