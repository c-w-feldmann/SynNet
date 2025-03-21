"""Generate synthetic trees."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from rdkit import RDLogger

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from synnet.data_generation.syntrees import SynTreeGenerator, SynTreeGeneratorPostProc
from synnet.utils.data_utils import ReactionSet, SyntheticTree

RDLogger.DisableLog("rdApp.*")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        default=os.getenv("BUILDING_BLOCKS_FILE"),
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        default=os.getenv("REACTION_TEMPLATES_FILE"),
        help="Input file with reaction templates as SMARTS(No header, one per line).",
    )
    parser.add_argument(
        "--rxn-collection-file",
        type=str,
        default=os.getenv("RXN_COLLECTION_FILE"),
        help="Input `*.json.gz` file with initialised reactions.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=os.getenv("SYNTREES_RAW_FILE"),
        help="Output file for the generated synthetic trees (*.json.gz)",
    )
    # Parameters
    parser.add_argument(
        "--number-syntrees",
        type=int,
        default=100,
        help="Number of SynTrees to generate.",
    )
    parser.add_argument(
        "--min-actions",
        type=int,
        default=1,
        help="Minimum number of actions per SynTree.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=7,
        help="Maximum number of actions per SynTree.",
    )

    # Processing
    parser.add_argument(
        "--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus"
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    if args.debug:
        st_logger = logging.getLogger("synnet.data_generation.syntrees")
        st_logger.setLevel("DEBUG")
        RDLogger.EnableLog("rdApp.*")

    # Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
    rxn_coll = (
        ReactionSet.load(args.rxn_collection_file) if args.rxn_collection_file else None
    )
    logger.info("Loaded building block & rxn-template assets.")

    # Init SynTree Generator
    logger.info("Start initializing SynTreeGenerator...")
    stgen = SynTreeGenerator(
        building_blocks=bblocks,
        rxn_templates=rxn_templates,
        rxn_collection=rxn_coll,
        verbose=args.verbose,
    )
    logger.info("Successfully initialized SynTreeGenerator.")

    # Generate synthetic trees
    logger.info(f"Start generation of {args.number_syntrees} SynTrees...")
    stgen_kwargs = {"max_depth": args.max_actions, "min_actions": args.min_actions}

    def stgen_with_fresh_seed(
        dummy: None, **stgen_kwargs: Any
    ) -> tuple[Optional[SyntheticTree], Optional[Exception]]:
        stgen.rng = np.random.default_rng()
        return stgen.generate_safe(
            max_depth=args.max_actions, min_actions=args.min_actions
        )

    func = partial(stgen_with_fresh_seed, stgen_kwargs=stgen_kwargs)
    with mp.Pool(args.ncpu) as pool:
        results = pool.map(func, range(args.number_syntrees))
    # results = chunked_parallel(range(args.number_syntrees),func,max_cpu=args.ncpu,verbose=True)

    syntrees, exit_codes = SynTreeGeneratorPostProc.parse_generate_safe(results)

    logger.info(f"SynTree generation completed. Exit codes: {exit_codes}")
    logger.info(f"Generated syntrees: {len(syntrees)}")

    summary_file = Path(args.output_file).parent / "results-summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing summary to {summary_file} .")
    summary = {
        "exit_codes": exit_codes,
        "args": vars(args),
        "stgen_kwargs": stgen_kwargs,
    }
    summary_file.write_text(json.dumps(exit_codes, indent=2))

    # Save synthetic trees on disk
    syntrees.save(args.output_file)

    logger.info(f"Generated syntrees: {len(syntrees)}")
    logger.info("Completed.")
