"""Filter Synthetic Trees."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

import abc
import json
from pathlib import Path
from typing import Any, Union

import numpy as np
from loguru import logger
from rdkit import Chem, RDLogger

from synnet.config import MAX_PROCESSES
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel

RDLogger.DisableLog("rdApp.*")


class Filter:
    @abc.abstractmethod
    def filter(self, st: SyntheticTree, **kwargs: Any) -> bool:
        """Filter a synthetic tree and return True if it passes the filter, else False.

        Abstract method. Must be implemented in the subclass.

        Parameters
        ----------
        st : SyntheticTree
            The synthetic tree to filter.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        bool
            True if the synthetic tree passes the filter, else False.
        """


class ValidRootMolFilter(Filter):
    def filter(self, st: SyntheticTree, **kwargs: Any) -> bool:
        """Filter for valid root molecules."""
        if not st.root or not st.root.smiles:
            return False
        return Chem.MolFromSmiles(st.root.smiles) is not None


class OracleFilter(Filter):
    def __init__(
        self,
        name: str = "qed",
        threshold: float = 0.5,
        rng: np.random.Generator = np.random.default_rng(42),
    ) -> None:
        super().__init__()
        from tdc import Oracle

        self.oracle_fct = Oracle(name=name)
        self.threshold = threshold
        self.rng = rng

    def _qed(self, st: SyntheticTree) -> bool:
        """Filter for molecules with a high qed."""
        if not st.root or not st.root.smiles:
            return False
        return self.oracle_fct(st.root.smiles) > self.threshold

    def _random(self, st: SyntheticTree) -> bool:
        """Filter molecules that fail the `_qed` filter; i.e. randomly select low qed molecules."""
        if not st.root or not st.root.smiles:
            return False
        return self.rng.random() < (self.oracle_fct(st.root.smiles) / self.threshold)

    def filter(self, st: SyntheticTree, **kwargs: Any) -> bool:
        if kwargs:
            raise ValueError(f"Unknown keyword arguments: {kwargs}")
        return self._qed(st) or self._random(st)


def filter_syntree(syntree: SyntheticTree) -> Union[SyntheticTree, int]:
    """Apply filters to `syntree` and return it, if all filters are passed. Else, return error code."""
    # Filter 1: Is root molecule valid?
    keep_tree = valid_root_mol_filter.filter(syntree)
    if not keep_tree:
        return -1

    # Filter 2: Is root molecule "pharmaceutically interesting?"
    keep_tree = interesting_mol_filter.filter(syntree)
    if not keep_tree:
        return -2

    # We passed all filters. This tree ascended to our dataset
    return syntree


if __name__ == "__main__":
    import argparse

    def get_args() -> argparse.Namespace:

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

        # Processing
        parser.add_argument(
            "--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus"
        )
        parser.add_argument("--verbose", default=False, action="store_true")
        return parser.parse_args()

    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load previously generated synthetic trees
    syntree_collection = SyntheticTreeSet().load(args.input_file)
    logger.info(
        f"Successfully loaded '{args.input_file}' with {len(syntree_collection)} syntrees."
    )

    # Filter trees
    # TODO: Move to src/synnet/data_generation/filters.py ?
    valid_root_mol_filter = ValidRootMolFilter()
    interesting_mol_filter = OracleFilter(threshold=0.5, rng=np.random.default_rng(42))

    syntrees = [s for s in syntree_collection.synthetic_tree_list if s is not None]

    logger.info(f"Start filtering {len(syntrees)} syntrees.")

    results: list[Union[SyntheticTree, int]]
    results = chunked_parallel(syntrees, filter_syntree, verbose=args.verbose)

    logger.info("Finished decoding.")

    # Handle results, most notably keep track of why we deleted the tree
    outcomes: dict[str, int] = {
        "invalid_root_mol": 0,
        "not_interesting": 0,
    }
    syntrees_filtered = []
    for res in results:
        if res == -1:
            outcomes["invalid_root_mol"] += 1
        if res == -2:
            outcomes["not_interesting"] += 1
        else:
            if isinstance(res, int):
                raise ValueError(f"Unknown error code: {res}")
            syntrees_filtered.append(res)

    logger.info("Successfully filtered syntrees.")

    out_folder = Path(args.output_file).parent
    out_folder.mkdir(parents=True, exist_ok=True)

    # Save filtered synthetic trees on disk
    SyntheticTreeSet(syntrees_filtered).save(args.output_file)
    logger.info(
        f"Successfully saved '{args.output_file}' with {len(syntrees_filtered)} syntrees."
    )

    # Save short summary file
    summary_file = Path(args.output_file).parent / "filter-summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(outcomes, indent=2))

    logger.info("Completed.")
