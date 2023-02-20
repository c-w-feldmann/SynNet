import logging
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from torch.utils.data.dataset import Dataset

from synnet.config import MAX_PROCESSES
from synnet.utils.datastructures import SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel

logger = logging.getLogger(__name__)


class SyntreeDataset(Dataset):
    def __init__(
        self,
        *,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntheticTreeSet],
        num_workers: int = MAX_PROCESSES,
        **kwargs,
    ):
        """Dataset for syntrees."""
        self.num_workers = num_workers

        if isfile := isinstance(dataset, Path) or isinstance(dataset, str):
            self.syntrees = SyntheticTreeSet.load(dataset).sts
            logger.info(f"Loaded dataset from file: {dataset}")
        elif isinstance(dataset, Iterable):
            self.syntrees = dataset
        elif isinstance(dataset, SyntheticTreeSet):
            self.syntrees = dataset.sts
        else:
            raise ValueError(f"dataset must be a Path, string or Iterable, not {type(dataset)}")

    def __len__(self):
        return len(self.syntrees)

    def __getitem__(self, index) -> SyntheticTree:
        return self.syntrees[index]

    def __repr__(self) -> str:
        return f"SyntreeDataset ({len(self)} syntrees)"


class SynTreeChopper:
    @staticmethod
    def chop_syntree(syntree: SyntheticTree) -> list[dict[str, Union[int, tuple[str, str, str]]]]:

        data = []
        target_mol = syntree.root.smiles
        # Recall: We can have at most 2 sub-trees, each with a root node.
        root_mol_1 = None
        root_mol_2 = None
        for i, action in enumerate(syntree.actions):
            target: int = action
            state: tuple[str, str, str] = (target_mol, root_mol_1, root_mol_2)

            if action == 3:
                reaction_id = None
                reactant_1 = None
                reactant_2 = None
            else:
                reaction_id = syntree.reactions[i].rxn_id
                reactant_1 = syntree.reactions[i].child[0]
                reactant_2 = (
                    syntree.reactions[i].child[1] if syntree.reactions[i].rtype == 2 else None
                )  # TODO: refactor datastructure: avoid need for ifs

            x = {
                "num_action": i,
                "target": target,
                "state": state,
                "reaction_id": reaction_id,
                "reactant_1": reactant_1,
                "reactant_2": reactant_2,
            }

            # The current action determines the state for the next iteration.
            # Note:
            #  - There can at most two "sub"syntrees.
            #  - There is always an "actively growing" and a "dangling" sub-syntree
            #  - We keep track of it:
            #    - root_mol_1 -> active branch
            #    - root_mol_2 -> dangling branch
            if action == 0:  # add: adds a new root_mol
                root_mol_2 = root_mol_1  # dangling branch is the previous active branch
                root_mol_1 = syntree.reactions[i].parent  # active branch
            elif action == 1:  # extend
                root_mol_2 = root_mol_2  # dangling, do not touch
                root_mol_1 = syntree.reactions[i].parent
            elif action == 2:  # merge
                root_mol_2 = None  # dangling branch is merged and can be reset
                root_mol_1 = syntree.reactions[i].parent
            elif action == 3:  # end
                pass
            else:
                raise ValueError(f"Action must be {0,1,2,3}, not {action}")

            data.append(x)
        return data


class RT1SyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Reactant 1** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: None = None,
        reactant_1_featurizer: None = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)
        self.featurizer = featurizer
        valid_actions = [0]  # "expand", "merge", and "end" have no reactant 1

        # Extract data
        chopped_syntrees = [self.chop_syntree(st) for st in self.syntrees]
        self.data = [
            elem
            for sublist in chopped_syntrees
            for elem in sublist
            if elem["target"] in valid_actions
        ]

        # Featurize data
        if featurizer:
            _features = chunked_parallel(
                [elem["state"] for elem in self.data],
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
                [elem["reactant_1"] for elem in self.data],
                reactant_1_featurizer.encode,
                max_cpu=num_workers,
                verbose=verbose,
            )
            self.targets = np.asarray(_targets).squeeze().astype("float32")  # (n, nbits')
        else:
            raise ValueError("No featurizer provided")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"RT1SyntreeDataset ({len(self)} datapoints)"


class RXNSyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Reaction** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: None = None,
        rxn_featurizer: None = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)
        self.featurizer = featurizer
        valid_actions = [0, 1, 2]  # "end" do not have reactions

        # Extract data
        chopped_syntrees = [self.chop_syntree(st) for st in self.syntrees]
        self.data = [
            elem
            for sublist in chopped_syntrees
            for elem in sublist
            if elem["target"] in valid_actions
        ]

        def _tupelize(elem: dict) -> tuple[Union[str, None]]:
            """Helper method to create a tuple for featurization."""
            return elem["state"] + (elem["reactant_1"],)

        _features = chunked_parallel(
            [_tupelize(elem) for elem in self.data],
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
            [rxn_featurizer.encode(elem["reaction_id"]) for elem in self.data]
        ).squeeze()  # (n, dimension_rxn_emb): z_rxn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"RXNSyntreeDataset ({len(self)} datapoints)"


class RT2SyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Reactant 2** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: None = None,
        rxn_featurizer: None = None,
        reactant_2_featurizer: None = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)
        self.featurizer = featurizer
        valid_actions = [0, 1]  # "end" and "merge" do not have 2nd reactants

        # Extract data
        chopped_syntrees = [self.chop_syntree(st) for st in self.syntrees]
        self.data = [
            elem
            for sublist in chopped_syntrees
            for elem in sublist
            if elem["target"] in valid_actions and elem["reactant_2"] is not None
        ]  # fmt: skip                         ^ exclude unimolecular reactions

        # Featurize data
        # x = z_target ⊕ z_state ⊕ z_rt1 ⊕ z_rxn
        # y = z_rt2
        def _tupelize(elem: dict) -> tuple[Union[str, None]]:
            """Helper method to create a tuple for featurization."""
            return elem["state"] + (elem["reactant_1"],)

        _features_mols = chunked_parallel(
            [_tupelize(elem) for elem in self.data],
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
            [rxn_featurizer.encode(elem["reaction_id"]) for elem in self.data]
        ).squeeze()  # (n, dimension_rxn_emb): z_rxn

        self.features = np.concatenate((_features_mols, _features_rxn), axis=1,).astype(
            "float32"
        )  # (n, 4*nbits + dimension_rxn_emb)

        # Targets
        _targets = chunked_parallel(
            [elem["reactant_1"] for elem in self.data],
            reactant_2_featurizer.encode,
            max_cpu=num_workers,
            verbose=verbose,
        )
        self.targets = np.asarray(_targets).squeeze().astype("float32")  # (n, nbits')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"RT2SyntreeDataset ({len(self)} datapoints)"


class ActSyntreeDataset(SyntreeDataset, SynTreeChopper):
    """SyntreeDataset for the **Action** network."""

    def __init__(
        self,
        dataset: Union[str, Path, Iterable[SyntheticTree], SyntreeDataset],
        featurizer: None = None,
        num_workers: int = MAX_PROCESSES,
        verbose: bool = False,
    ):
        """ """
        # Init superclass
        super().__init__(dataset=dataset, num_workers=num_workers)
        self.featurizer = featurizer

        # Extract data
        # For the ACT network the problem is classification.
        chopped_syntrees = [self.chop_syntree(st) for st in self.syntrees]
        self.data = [elem for sublist in chopped_syntrees for elem in sublist]

        # Featurize data
        if featurizer:
            _features = chunked_parallel(
                [elem["state"] for elem in self.data],
                featurizer.encode_tuple,
                max_cpu=num_workers,
                verbose=verbose,
            )
            _features = np.asarray(_features)
            shape = _features.shape
            self.features = _features.reshape((-1, shape[-2] * shape[-1])).astype(
                "float32"
            )  # (num_states, 3*nbits for MorganFP OR 'nbits' for drfp)
        else:
            raise ValueError("No featurizer provided")

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.features[idx], np.asarray(self.data[idx]["target"], dtype="int32")

    def __repr__(self) -> str:
        return f"ActSyntreeDataset ({len(self)} datapoints)"
