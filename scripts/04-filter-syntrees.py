"""Filter Synthetic Trees.
"""

import json
import logging
from pathlib import Path

import numpy as np
from rdkit import RDLogger

from synnet.config import MAX_PROCESSES
from synnet.utils.datastructures import SyntheticTreeSet
from synnet.utils.filters import FILTERS, calc_metrics_on_syntree_collection

logger = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.*")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for the filtered generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for the filtered generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="qed + random",
        choices=FILTERS.keys(),
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

    # Load previously generated synthetic trees
    syntree_collection = SyntheticTreeSet().load(args.input_file)
    logger.info(f"Successfully loaded '{args.input_file}' with {len(syntree_collection)} syntrees.")

    # Filter synthetic trees
    rng = np.random.default_rng(42)
    THRESHOLD = 0.5  # for QED filter

    df = calc_metrics_on_syntree_collection(syntree_collection)
    df["random"] = rng.random(len(df))

    query = FILTERS[args.filter]
    logger.info(f"Filtering syntrees with query: {query}")
    filtered_df = df.query(query)

    logger.info(f"Successfully filtered syntrees.")
    logger.info(
        f"Retained {len(filtered_df)} syntrees out of {len(df)} ({len(filtered_df)/len(df)*100:.2f}%)"
    )

    out_folder = Path(args.output_file).parent
    out_folder.mkdir(parents=True, exist_ok=True)

    # Save filtered synthetic trees on disk
    SyntheticTreeSet(filtered_df["syntrees"].values.tolist()).save(args.output_file)
    logger.info(f"Successfully saved '{args.output_file}' with {len(filtered_df)} syntrees.")

    # Save short summary file
    _summary = {
        "input_file": {"metadata": syntree_collection.metadata},
        "output_file": args.output_file,
        "filter": {
            "label": "qed + random",
            "query": query,
            "threshold": THRESHOLD,
            "num_syntrees": len(filtered_df),
        },
        "pandas_describe": filtered_df.describe().to_dict(),
    }
    summary_file = Path(args.output_file).parent / "filter-summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(_summary, indent=2))
    logger.info(f"Successfully saved '{summary_file}'.")

    logger.info(f"Completed.")
