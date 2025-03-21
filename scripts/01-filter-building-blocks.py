"""Filter out building blocks that cannot react with any template."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

import argparse
import json

from loguru import logger
from rdkit import RDLogger

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilterHeuristics,
    BuildingBlockFilterMatchRxn,
    ReactionTemplateFileHandler,
)
from synnet.utils.data_utils import ReactionSet

RDLogger.DisableLog("rdApp.*")


def get_args() -> argparse.Namespace:
    """Parse input arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="File with SMILES strings (DataFrame with `SMILES` column).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        help="Input file with reaction templates as SMARTS (No header, one per line).",
    )
    parser.add_argument(
        "--output-bblock-file",
        type=str,
        help="Output file for the filtered building-blocks.",
    )
    parser.add_argument(
        "--output-rxns-collection-file",
        type=str,
        help="Output file for the collection of reactions matched with building-blocks.",
    )
    # Processing
    parser.add_argument(
        "--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus"
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def filter_building_blocks() -> None:
    """Filter building blocks that cannot react with any template."""
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # 1. Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    # 2. Filter
    #   building blocks on heuristics
    filtered_bblocks = BuildingBlockFilterHeuristics(
        verbose=args.verbose
    ).filter_to_list(bblocks)

    #   building blocks that cannot react with any template
    filtered_bblocks_list, reactions = BuildingBlockFilterMatchRxn().filter(
        filtered_bblocks, rxn_templates, ncpu=args.ncpu, verbose=args.verbose
    )

    # 3. Save
    #   filtered building blocks
    BuildingBlockFileHandler().save(args.output_bblock_file, filtered_bblocks_list)

    #   initialized reactions (these are initialized with available reactants)
    ReactionSet(reactions).save(
        args.output_rxns_collection_file, skip_without_building_block=False
    )

    logger.info("Completed.")


if __name__ == "__main__":
    filter_building_blocks()
