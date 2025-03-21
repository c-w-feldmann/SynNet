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
    """SyntreeDataset."""

    syntree_set: SyntheticTreeSet

    def __init__(
        self,
        *,
        dataset: Union[PathType, Iterable[SyntheticTree], SyntreeDataset],
        num_workers: int = MAX_PROCESSES,
    ) -> None:
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
        """Return the number of syntrees in the dataset."""
        return len(self.syntree_set)

    def __getitem__(self, index: int) -> SyntheticTree:
        """Return the syntree at the given index."""
        return self.syntree_set[index]

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"SyntreeDataset ({len(self)} syntrees)"
