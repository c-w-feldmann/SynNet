from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
from numpy import typing as npt
from torch.utils.data.dataset import Dataset

from synnet.config import MAX_PROCESSES
from synnet.data_generation.syntrees import MorganFingerprintEncoder
from synnet.utils.custom_types import PathType
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel


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


class ActSyntreeDataset(SyntreeDataset):
    """SyntreeDataset."""

    featurizer: MorganFingerprintEncoder
    features: npt.NDArray[np.float_]

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: MorganFingerprintEncoder,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        """ """
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)
        self.featurizer = featurizer

        # Extract data
        # For the ACT network the problem is classification.
        chopped_syntrees = [
            self.chop_syntree(st) for st in self.syntree_set.synthetic_tree_list
        ]
        self.data = [elem for sublist in chopped_syntrees for elem in sublist]

        # Featurize data
        if self.featurizer is not None:
            # New: use `chunked_parallel()`
            _features = chunked_parallel(
                [elem["state"] for elem in self.data],
                self.featurizer.encode_batch,
                max_cpu=num_workers,
                verbose=verbose,
            )
            self.features = (
                np.asarray(_features)
                .reshape((-1, 3 * self.featurizer.nbits))
                .astype("float32")
            )  # (num_states, 3*nbits)
        else:
            raise ValueError("No featurizer provided")

    @staticmethod
    def chop_syntree(syntree: SyntheticTree) -> list[dict[str, Any]]:
        data = []
        if syntree.root is None:
            raise ValueError("Syntree must have a root node")
        target_mol = syntree.root.smiles
        # Recall: We can have at most 2 sub-trees, each with a root node.
        root_mol_1 = None
        root_mol_2 = None
        for i, action in enumerate(syntree.actions):
            state: tuple[str, Optional[str], Optional[str]] = (
                target_mol,
                root_mol_1,
                root_mol_2,
            )
            target: int = action
            x = {"target": target, "state": state, "num_action": i}

            # The current action determines the state for the next iteration.
            # Note:
            #  - There can at most two "sub"syntrees.
            #  - There is always an "actively growing" and a "dangling" sub-syntree
            #  - We keep track of it:
            #   - root_mol_1 -> active branch
            #   - root_mol_2 -> dangling branch
            if action == 0:  # add : adds a new root_mol
                root_mol_2 = root_mol_1  # dangling branch
                root_mol_1 = syntree.reactions[i].parent  # active branch
            elif action == 1:  # extend
                root_mol_1 = syntree.reactions[i].parent
                root_mol_2 = root_mol_2  # dangling, do not touch
            elif action == 2:  # merge
                root_mol_1 = syntree.reactions[i].parent
                root_mol_2 = None  # dangling branch is merged and can be reset
            elif action == 3:  # end
                pass
            else:
                raise ValueError(f"Action must be {0,1,2,3}, not {action}")
            data.append(x)
        return data

    def __len__(self) -> int:
        """__len__."""
        return len(self.data)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
        """Get ml inputs and targets.

        TODO: Harmonize typing with `SyntreeDataset.__getitem__`.

        Parameters
        ----------
        idx : int
            Index of the data point to retrieve.

        Returns
        -------
        tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]
            Features and targets.
        """
        return self.features[idx], np.asarray(self.data[idx]["target"], dtype="int32")
