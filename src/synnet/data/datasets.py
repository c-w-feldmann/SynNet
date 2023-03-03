import enum
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from torch.utils.data.dataset import Dataset

from synnet.config import MAX_PROCESSES
from synnet.data_generation.syntrees import Encoder
from synnet.utils.datastructures import SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel

logger = logging.getLogger(__name__)


class SyntreeDataset(Dataset):
    """torch `Dataset` for syntrees."""

    def __init__(
        self,
        *,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntheticTreeSet],
        num_workers: int = MAX_PROCESSES,
        **kwargs,
    ):
        self.num_workers = num_workers

        if isinstance(dataset, Path) or isinstance(dataset, str):
            self.syntrees = SyntheticTreeSet.load(dataset).sts
            logger.info(f"Loaded dataset from file: {dataset}")
        elif isinstance(dataset, Iterable):
            self.syntrees = dataset
        elif isinstance(dataset, SyntheticTreeSet):
            self.syntrees = dataset.sts
        else:
            raise ValueError(f"dataset must be a Path, string or Iterable, not {type(dataset)}")

    def __len__(self) -> int:
        return len(self.syntrees)

    def __getitem__(self, index) -> SyntheticTree:
        return self.syntrees[index]

    def __repr__(self) -> str:
        return f"SyntreeDataset ({len(self)} syntrees)"


class Action(enum.IntEnum):  # TODO: refactor everywhere
    """Actions that can be taken on a syntree"""

    add = 0
    expand = 1
    merge = 2
    end = 3


@dataclass
class SynTreeChunk:
    """A chunk of a syntree."""

    num_action: int  # = t, i.e. the number of actions taken so far
    action: int
    state: tuple[str, str, str]
    reaction_id: int
    reactant_1: str
    reactant_2: Optional[str]


class SynTreeChopper:
    """Chops a syntree into chunks."""

    @staticmethod
    def chop_batch(syntrees: Iterable[SyntheticTree]) -> list[SynTreeChunk]:
        """Chops a batch of syntrees into chunks (flattens the list of chunks)"""
        return [chunk for syntree in syntrees for chunk in SynTreeChopper.chop(syntree)]

    @staticmethod
    def chop(syntree: SyntheticTree) -> list[SynTreeChunk]:
        """Chops a syntree into chunks."""

        chunks = []
        target_mol = syntree.root.smiles
        # Recall: We can have at most 2 sub-trees, each with a root node.
        root_mol_active = None
        root_mol_inactive = None
        for i, action in enumerate(syntree.actions):
            state: tuple[str, str, str] = (target_mol, root_mol_active, root_mol_inactive)
            _action_name = Action(action).name

            if _action_name == "end":
                reaction_id = None
                reactant_1 = None
                reactant_2 = None
            else:
                reaction_id = syntree.reactions[i].rxn_id
                reactant_1 = syntree.reactions[i].child[0]
                reactant_2 = (
                    syntree.reactions[i].child[1] if syntree.reactions[i].rtype == 2 else None
                )  # TODO: refactor datastructure: avoid need for ifs

            chunk = SynTreeChunk(
                num_action=i,
                action=action,
                state=state,
                reaction_id=reaction_id,
                reactant_1=reactant_1,
                reactant_2=reactant_2,
            )

            # The current action determines the state for the next iteration.
            # Note:
            #  - There can at most two "sub"syntrees.
            #  - There is always an "active" and a "inactive" sub-syntree
            #  - The "active" sub-syntree is the one actively being expanded
            if _action_name == "add":
                root_mol_inactive = root_mol_active  # active branch becomes inactive
                root_mol_active = syntree.reactions[i].parent  # product = root mol of active branch
            elif _action_name == "expand":
                root_mol_inactive = root_mol_inactive  # inactive branch remains inactive
                root_mol_active = syntree.reactions[i].parent  # product = root mol of active branch
            elif _action_name == "merge":
                root_mol_inactive = None  # inactive branch is reset
                root_mol_active = syntree.reactions[i].parent  # merge-rxn product -> active branch
            elif _action_name == "end":
                pass
            else:
                raise ValueError(f"Action must be {0,1,2,3}, not {action}")

            chunks.append(chunk)
        return chunks


class RT1SyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Reactant 1** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: Encoder = None,
        featurizer_reactant_1: Encoder = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)

        # Featurizers
        self.featurizer = featurizer

        # Syntree chunks (filtered)
        def _filter(chunk: SynTreeChunk) -> bool:
            return chunk.action in [0]  # "expand", "merge", and "end" have no reactant 1

        chunks = self.chop_batch(self.syntrees)
        self.chunks = [chunk for chunk in chunks if _filter(chunk)]

        # Featurize data
        # x = z_state
        # y = z_reactant_1
        if featurizer:
            _features = chunked_parallel(
                [chunk.state for chunk in self.chunks],
                featurizer.encode_tuple,
                max_cpu=num_workers,
                verbose=verbose,
            )
            _features = np.asarray(_features)
            shape = _features.shape
            self.features = (
                np.asarray(_features).reshape((-1, shape[-2] * shape[-1])).astype("float32")
            )  # (num_states, 3*nbits for MorganFP OR 'nbits' for drfp)

            _targets = chunked_parallel(
                [chunk.reactant_1 for chunk in self.chunks],
                featurizer_reactant_1.encode,
                max_cpu=num_workers,
                verbose=verbose,
            )
            self.targets = np.asarray(_targets).squeeze().astype("float32")  # (n, nbits')
        else:
            warnings.warn(f"No featurizer provided for {self.__class__.__name__}.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"RT1SyntreeDataset ({len(self)} datapoints)"


class RXNSyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Reaction** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: Encoder = None,
        featurizer_rxn: Encoder = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)

        # Featurizers
        self.featurizer = featurizer

        # Syntree chunks (filtered)
        def _filter(chunk: SynTreeChunk) -> bool:
            return chunk.action in [0, 1, 2]  # "end" do not have reactions

        chunks = self.chop_batch(self.syntrees)
        self.chunks = [chunk for chunk in chunks if _filter(chunk)]

        # Featurize data
        # x = z_state ⊕ z_rt1
        # y = z_rxn
        if featurizer:

            def _tupelize(chunk: SynTreeChunk) -> tuple[Union[str, None]]:
                """Helper method to create a tuple for featurization."""
                return chunk.state + (chunk.reactant_1,)

            _features = chunked_parallel(
                [_tupelize(chunk) for chunk in self.chunks],
                featurizer.encode_tuple,
                max_cpu=num_workers,
                verbose=verbose,
            )
            _features = np.asarray(_features)
            shape = _features.shape
            self.features = self.features = (
                np.asarray(_features).reshape((-1, shape[-2] * shape[-1])).astype("float32")
            )  # (num_states, 4*nbits for MorganFP OR 'nbits' for drfp)

            self.targets = np.asarray(
                [featurizer_rxn.encode(chunk.reaction_id) for chunk in self.chunks]
            ).squeeze()  # (n, dimension_rxn_emb): z_rxn
        else:
            warnings.warn(f"No featurizer provided for {self.__class__.__name__}.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"RXNSyntreeDataset ({len(self)} datapoints)"


class RT2SyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Reactant 2** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: Encoder = None,
        featurizer_rxn: Encoder = None,
        featurizer_reactant_2: Encoder = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)

        # Featurizers
        self.featurizer = featurizer

        # Syntree chunks (filtered)
        def _filter(chunk: SynTreeChunk) -> bool:
            is_valid_action = chunk.action in [0, 1]  # "end" & "merge" do not have 2nd reactants
            has_2nd_reactant = chunk.reactant_2 is not None
            return is_valid_action and has_2nd_reactant

        chunks = self.chop_batch(self.syntrees)
        self.chunks = [chunk for chunk in chunks if _filter(chunk)]

        # Featurize data
        # x = z_state ⊕ z_rt1 ⊕ z_rxn
        # y = z_rt2
        if featurizer:

            def _tupelize(chunk: dict) -> tuple[Union[str, None]]:
                """Helper method to create a tuple for featurization."""
                return chunk.state + (chunk.reactant_1,)

            _features_mols = chunked_parallel(
                [_tupelize(chunk) for chunk in self.chunks],
                featurizer.encode_tuple,
                max_cpu=num_workers,
                verbose=verbose,
            )
            _features_mols = np.asarray(_features_mols)
            shape = _features_mols.shape
            _features_mols = (
                np.asarray(_features_mols).reshape((-1, shape[-2] * shape[-1])).astype("float32")
            )  # (n, d): z_(target ⊕ state) ⊕ z_rt1

            _features_rxn = np.asarray(
                [featurizer_rxn.encode(chunk.reaction_id) for chunk in self.chunks]
            ).squeeze()  # (n, dimension_rxn_emb): z_rxn

            self.features = np.concatenate((_features_mols, _features_rxn), axis=1,).astype(
                "float32"
            )  # (n, 4*nbits + dimension_rxn_emb)

            # Targets
            _targets = chunked_parallel(
                [chunk.reactant_1 for chunk in self.chunks],
                featurizer_reactant_2.encode,
                max_cpu=num_workers,
                verbose=verbose,
            )
            self.targets = np.asarray(_targets).squeeze().astype("float32")  # (n, nbits')
        else:
            warnings.warn(f"No featurizer provided for {self.__class__.__name__}.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"RT2SyntreeDataset ({len(self)} datapoints)"


class ActSyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Action** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: Encoder = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        """ """
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)
        self.featurizer = featurizer

        # Extract data
        chunks = self.chop_batch(self.syntrees)
        self.chunks = chunks

        # Featurize data
        if featurizer:
            _features = chunked_parallel(
                [chunk.state for chunk in self.chunks],
                featurizer.encode_tuple,
                max_cpu=num_workers,
                verbose=verbose,
            )
            _features = np.asarray(_features)
            shape = _features.shape
            self.features = _features.reshape((-1, shape[-2] * shape[-1])).astype(
                "float32"
            )  # (num_states, 3*nbits for MorganFP OR 'nbits' for drfp)

            # region inject one-hot-encoding of num_action
            if os.environ.get("SYNNET_ACT_POS"):  # feature flag for amateurs
                print("😊Injecting one-hot-encoding of num_action into features.")
                MAX_DEPTH = max([chunk.num_action for chunk in self.chunks])
                print(f"😊Syntrees have a maximum depth of {MAX_DEPTH}.")
                from synnet.data_generation.syntrees import OneHotEncoder

                onehot = OneHotEncoder(MAX_DEPTH)
                _depths = np.asarray([onehot.encode(chunk.num_action - 1) for chunk in self.chunks])
                _depths = _depths.astype("float32").squeeze()
                self.features = np.concatenate((self.features, _depths), axis=1)
            # endregion
            self.targets = np.asarray([chunk.action for chunk in self.chunks]).astype("int32")
        else:
            raise ValueError("No featurizer provided")

    def __len__(self):
        """__len__"""
        return len(self.chunks)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"ActSyntreeDataset ({len(self)} datapoints)"
