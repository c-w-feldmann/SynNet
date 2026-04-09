"""Filter the synthetic trees."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from loguru import logger as logging
from torch.utils.data.dataset import Dataset

from synnet.config import MAX_PROCESSES
from synnet.utils.custom_types import PathType
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet


class SyntreeDataset(Dataset):  # type: ignore[type-arg]
    """Dataset wrapper for synthetic trees."""

    syntree_set: SyntheticTreeSet

    def __init__(
        self,
        *,
        dataset: PathType | Iterable[SyntheticTree] | SyntreeDataset,
        num_workers: int = MAX_PROCESSES,
    ) -> None:
        """Initialize a synthetic-tree dataset.

        Parameters
        ----------
        dataset : PathType | Iterable[SyntheticTree] | SyntreeDataset
            Data source as file path, iterable of trees, or existing dataset.
        num_workers : int, default=MAX_PROCESSES
            Number of workers used by downstream dataloaders.

        """
        if isinstance(dataset, (str, Path)):
            self.syntree_set = SyntheticTreeSet.load(dataset)
            logging.info(f"Loaded from file: {dataset}")
        elif isinstance(dataset, Iterable):
            tree_list: list[SyntheticTree] = list(dataset)  # type: ignore # potential mypy bug
            self.syntree_set = SyntheticTreeSet(tree_list)
        elif isinstance(dataset, SyntreeDataset):
            self.syntree_set = dataset.syntree_set
        else:
            raise ValueError(
                f"dataset must be a Path, string or Iterable, not {type(dataset)}"
            )
        self.num_workers = num_workers

    def __len__(self) -> int:
        """Return the number of synthetic trees in the dataset.

        Returns
        -------
        int
            Number of stored synthetic trees.

        """
        return len(self.syntree_set)

    def __getitem__(self, index: int) -> SyntheticTree:
        """Return the synthetic tree at the given index.

        Parameters
        ----------
        index : int
            Position of the requested item.

        Returns
        -------
        SyntheticTree
            Synthetic tree at ``index``.

        """
        return self.syntree_set[index]

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            Human-readable dataset summary.

        """
        return f"SyntreeDataset ({len(self)} syntrees)"
