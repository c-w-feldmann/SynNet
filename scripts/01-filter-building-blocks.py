"""Filter out building blocks that cannot react with any template.
"""
import logging

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
logger = logging.getLogger(__file__)
import json


def get_args():
    import argparse

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
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # 1. Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    # 2. Filter
    #   building blocks on heuristics
    filtered_bblocks = BuildingBlockFilterHeuristics.filter(bblocks, verbose=args.verbose)

    #   building blocks that cannot react with any template
    filtered_bblocks, reactions = BuildingBlockFilterMatchRxn.filter(
        filtered_bblocks, rxn_templates, ncpu=args.ncpu, verbose=args.verbose
    )

    # 3. Save
    #   filtered building blocks
    BuildingBlockFileHandler().save(args.output_bblock_file, filtered_bblocks)

    #   initialized reactions (these are initialized with available reactants)
    ReactionSet(reactions).save(args.output_rxns_collection_file)

    logger.info("Completed.")
