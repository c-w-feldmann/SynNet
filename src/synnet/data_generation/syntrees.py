"""syntrees
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse
from tqdm import tqdm

from synnet.config import MAX_PROCESSES
from synnet.data_generation.exceptions import (
    MaxNumberOfActionsError,
    NoBiReactionAvailableError,
    NoMergeReactionPossibleError,
    NoReactantAvailableError,
    NoReactionAvailableError,
    NoReactionPossibleError,
)
from synnet.utils.data_utils import (
    NodeRxn,
    Reaction,
    ReactionSet,
    SyntheticTree,
    SyntheticTreeSet,
)

logger = logging.getLogger(__name__)


class SynTreeGenerator:
    """Generates synthetic trees by randomly applying reactions to building blocks."""

    ACTIONS: dict[int, str] = {
        i: action for i, action in enumerate("add expand merge end".split())
    }
    reaction_set: ReactionSet

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        rxn_collection: Optional[ReactionSet] = None,
        rng: np.random.Generator = np.random.default_rng(),
        processes: int = MAX_PROCESSES,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initializes a `SynTreeGenerator`.

        Parameters
        ----------
        building_blocks : list[str]
            List of building blocks.
        rxn_templates : list[str]
            List of reaction templates.
        rxn_collection : Optional[ReactionSet], optional
            Pre-initialised reaction collection, by default None
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()
            Note: When generating syntrees with mp provide a fresh seed!
        processes : int, optional
            Number of processes to use, by default MAX_PROCESSES
        verbose : bool, optional
            Whether to show progress bars, by default False
        debug : bool, optional
            Whether to show debug messages, by default False
        """
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates
        self.reaction_set = ReactionSet(
            [Reaction(template=tmplt) for tmplt in rxn_templates]
        )
        self.rng = rng
        self.IDX_RXNS = np.arange(len(self.reaction_set))
        self.processes = processes
        self.verbose = verbose
        if not verbose:
            logger.setLevel("CRITICAL")  # dont show error msgs
        if debug:
            logger.setLevel("DEBUG")

        if rxn_collection is None:
            # We need to initialise all reactions with their applicable building blocks.
            # This is a time intensive task, so better to init with `rxn_collection`.
            self._init_rxns_with_reactants()
        else:
            logger.info("Using pre-initialised reaction collection.")
            self.reaction_set = rxn_collection
            self.rxns_initialised = True

    def __match_mp(self) -> Self:
        # TODO: refactor / merge with `BuildingBlockFilter`
        # TODO: Rename `ReactionSet` -> `ReactionCollection` (same for `SyntheticTreeSet`)
        from functools import partial

        from pathos import multiprocessing as mp

        def __match(_rxn: Reaction, bblocks: list[str]) -> Reaction:
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, bblocks=self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            rxns = pool.map(func, self.reaction_set)

        self.reaction_set = rxns
        return self

    def _init_rxns_with_reactants(self) -> Self:
        """Initializes a `Reaction` with a list of possible reactants.

        Info: This can take a while for lots of possible reactants."""
        if self.processes == 1:
            rxns = tqdm(self.reaction_set) if self.verbose else self.reaction_set
            self.reaction_set = ReactionSet(
                [rxn.set_available_reactants(self.building_blocks) for rxn in rxns]
            )
        else:
            self.__match_mp()

        self.rxns_initialised = True
        return self

    def _sample_molecule(self) -> str:
        """Sample a molecule."""
        idx = self.rng.choice(len(self.building_blocks))
        smiles = self.building_blocks[idx]

        logger.debug(f"    Sampled molecule: {smiles}")
        return smiles

    def _find_rxn_candidates(
        self, smiles: str, raise_exc: bool = True
    ) -> npt.NDArray[np.bool_]:
        """Find reactions which reactions have `smiles` as reactant."""
        rxn_mask = [rxn.is_reactant(smiles) for rxn in self.reaction_set.rxns]

        if raise_exc and not any(
            rxn_mask
        ):  # Do not raise exc when checking if two mols can react
            raise NoReactionAvailableError(
                f"Cannot find a reaction for reactant: {smiles}."
            )
        return np.asarray(rxn_mask)

    def _sample_rxn(
        self, mask: Optional[npt.NDArray[np.bool_]] = None
    ) -> Tuple[Reaction, int]:
        """Sample a reaction by index."""
        if mask is None:
            irxn_mask = self.IDX_RXNS  # all reactions are possible
        else:
            irxn_mask = self.IDX_RXNS[mask]

        idx = self.rng.choice(irxn_mask)
        rxn = self.reaction_set[idx]

        logger.debug(
            f"    Sampled {'uni' if rxn.num_reactant == 1 else 'bi'}-molecular reaction with id {idx:d}"
        )
        return rxn, idx

    def _expand(
        self, reactant_1: str, raise_exc: bool = True
    ) -> tuple[str, Optional[str], Optional[str], int]:
        """Expand a sub-tree from one molecule.
        This can result in uni- or bimolecular reaction."""

        # Identify applicable reactions
        rxn_mask = self._find_rxn_candidates(reactant_1)

        # Sample reaction (by index)
        rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)
        available_reactants = rxn.available_reactants
        if not available_reactants:
            raise NoReactantAvailableError(
                f"No reactant available for reaction (ID: {idx_rxn}). Present reactant: {reactant_1}."
            )

        # Sample 2nd reactant
        if rxn.num_reactant == 1:
            reactant_2: Optional[str] = None
        else:
            # Sample a molecule from the available reactants of this reaction
            # That is, for a reaction A + B -> C,
            #   - determine if we have "A" or "B"
            #   - then sample "B" (or "A")
            if rxn.is_reactant_first(reactant_1):
                reactant_2_candidate_list = available_reactants[1]
            else:
                reactant_2_candidate_list = available_reactants[0]

            if not reactant_2_candidate_list:
                raise NoReactantAvailableError(
                    f"No reactant available for bimolecular reaction (ID: {idx_rxn}). Present reactant: {reactant_1}."
                )

            reactant_2 = self.rng.choice(reactant_2_candidate_list)
            logger.debug(f"    Sampled second reactant: {reactant_2}")

        # Run reaction
        reactants = (reactant_1, reactant_2)
        product = rxn.run_reaction(reactants)
        if raise_exc and product is None:
            raise NoReactionPossibleError(
                f"Reaction (ID: {idx_rxn}) not possible with: `{reactant_1} + {reactant_2}`."
            )
        return reactant_1, reactant_2, product, idx_rxn

    def _merge(self, syntree: SyntheticTree) -> Tuple[str, Optional[str], str, int]:
        """Merge the two root mols in the `SyntheticTree`"""
        # Identify suitable rxn
        r1: Optional[str]
        r2: Optional[str]
        r1, r2 = syntree.get_state()
        if r1 is None:
            raise NoMergeReactionPossibleError("Cannot merge empty tree.")
        rxn_mask = self._get_rxn_mask((r1, r2))
        # Sample reaction
        rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)
        # Run reaction
        p = rxn.run_reaction((r1, r2))
        if p is None:
            raise NoMergeReactionPossibleError(
                f"Reaction (ID: {idx_rxn}) not possible with: {r1} + {r2}."
            )
        return r1, r2, p, idx_rxn

    def _get_action_mask(self, syntree: SyntheticTree) -> npt.NDArray[np.bool_]:
        """Get a mask of possible action for a SyntheticTree"""
        # Recall: (Add, Expand, Merge, and End)
        can_add = False
        can_merge = False
        can_expand = False
        can_end = False

        state = syntree.get_state()
        if state[0] is None and state[1] is None:  # empty tree
            can_add = True
        elif state[0] is not None and state[1] is None:
            can_add = True
            can_expand = True
            can_end = True
        elif state[0] is not None and state[1] is not None:
            can_expand = True
            can_merge = any(self._get_rxn_mask(state, raise_exc=False))
        else:
            raise AssertionError("Invalid state.")

        # Special cases.
        # Case: Syntree is 1 update apart from its max size, only allow to end it.
        if (
            state[0] is not None
            and state[1] is None
            and (syntree.num_actions == self.max_depth - 1)
        ):
            logger.debug(
                "  Overriding action space to only allow action=end."
                + f"(1, {syntree.num_actions=}, {self.max_depth=})"
            )
            can_add, can_merge, can_expand = False, False, False
            can_end = True
        elif (
            state[0] is not None
            and state[1] is not None
            and (syntree.num_actions == self.max_depth - 2)
        ):
            # ATTN: This might result in an Exception,
            #       i.e. when no rxn template matches or the product is invalid etc.
            logger.debug(
                "  Overriding action space to forcefully merge trees."
                + f"(2, {syntree.num_actions=}, {self.max_depth=})"
            )
            can_add, can_expand, can_end = False, False, False
            can_merge = True
        elif (
            state[0] is not None
            and state[1] is None
            and (syntree.num_actions == self.max_depth - 3)
        ):
            # Handle case for max_depth=6 and [0, 1, 1, 1, 0, 2, 3] ?
            #                                              ^-- prevent this
            pass
        # Case: Syntree is 2 updates away from its max size, only allow to merge it.
        if syntree.num_actions < self.min_actions:
            can_end = False
        return np.array((can_add, can_expand, can_merge, can_end), dtype=bool)

    def _get_rxn_mask(
        self, reactants: tuple[Optional[str], Optional[str]], raise_exc: bool = True
    ) -> npt.NDArray[np.bool_]:
        """Get a mask of possible reactions for the two reactants."""
        # First: Identify bi-molecular reactions
        masks_bimol = [
            rxn.num_reactant == 2 for rxn in self.reaction_set.rxns
        ]  # TODO: Cache?
        # Second: Check if reactants match template in correct or reversed order, i.e.
        #         check if (r1->position1 & r2->position2) "ordered"
        #         or       (r1->position2 & r2->position1) "reversed"
        r1, r2 = reactants
        masks_r1 = [
            (
                (rxn.is_reactant_first(r1), rxn.is_reactant_second(r1))
                if is_bi
                else (False, False)
            )
            for is_bi, rxn in zip(masks_bimol, self.reaction_set.rxns)
        ]
        masks_r2 = [
            (
                (rxn.is_reactant_first(r2), rxn.is_reactant_second(r2))
                if is_bi
                else (False, False)
            )
            for is_bi, rxn in zip(masks_bimol, self.reaction_set.rxns)
        ]

        # Check if reactants match template the ordered or reversed way
        arr = np.array(
            (masks_r1, masks_r2)
        )  # (nReactant, nReaction, first-second-position)
        arr = arr.swapaxes(
            0, 1
        )  # view:         (nReaction, nReactant, first-second-position)
        canReactOrdered = np.trace(arr, axis1=1, axis2=2) > 1  # (nReaction,)
        canReactReversed = (
            np.flip(arr, axis=1).trace(axis1=1, axis2=2) > 1
        )  # (nReaction,)
        mask = np.logical_or(canReactOrdered, canReactReversed).tolist()

        if raise_exc and not any(mask):
            raise NoBiReactionAvailableError(f"No reaction available for {reactants}.")
        return np.asarray(mask)

    def _sample_action(self, syntree: SyntheticTree) -> int:
        """Samples an action conditioned on the state of the `SyntheticTree`"""
        p_action = self.rng.random((1, 4))  # (1,4)
        action_mask = self._get_action_mask(syntree)  # (1,4)
        act = int(np.argmax(p_action * action_mask))  # (1,)

        logger.debug(f"  Sampled action: {self.ACTIONS[act]}")
        return act

    def generate(self, max_depth: int = 8, min_actions: int = 1) -> SyntheticTree:
        """Generate a syntree by random sampling."""
        assert min_actions < max_depth, "min_actions must be smaller than max_depth."
        assert (
            max_depth > 1
        ), "max_actions must be larger than 1. (smallest treee is [`add`,`end`]"

        logger.debug(
            f"Starting synthetic tree generation with {min_actions=} and {max_depth=} "
        )
        self.max_depth = max_depth  # TODO: rename to reflect "number of actions"
        self.min_actions = min_actions

        # Init
        syntree = SyntheticTree()
        recent_mol: Optional[str] = None
        action = None
        for i in range(max_depth + 1):
            logger.debug(
                f"Iter {i} | {syntree.depth=} | num_actions={syntree.actions.__len__()} "
            )

            # Sample action
            act = self._sample_action(syntree)
            action = self.ACTIONS[act]

            if action == "end":
                r1, r2, p, idx_rxn = None, None, None, -1
            elif action == "expand":
                # Expand this subtree: reaction, (reactant2), run it.
                if recent_mol is None:
                    raise AssertionError("Cannot expand without recent molecule.")
                r1, r2, p, idx_rxn = self._expand(recent_mol)
            elif action == "add":
                # Add a new subtree: sample first reactant, then expand from there.
                mol = self._sample_molecule()
                r1, r2, p, idx_rxn = self._expand(mol)
            elif action == "merge":
                # Merge two subtrees: sample reaction, run it.
                r1, r2, p, idx_rxn = self._merge(syntree)
            else:
                raise ValueError(f"Invalid action {action}")

            # Prepare next iteration
            logger.debug(f"    Ran reaction {r1} + {r2} -> {p}")

            recent_mol = p

            # Update tree
            syntree.update(
                action=int(act), rxn_id=int(idx_rxn), mol1=r1, mol2=r2, mol_product=p
            )
            logger.debug("SynTree updated.")
            if action == "end":
                break

        if syntree.num_actions > max_depth and not action == "end":
            raise MaxNumberOfActionsError(
                f"Maximum number of actions exceeded. ({syntree.actions=}>{max_depth})."
            )
        logger.debug("SynTree completed.")
        return syntree

    def generate_safe(
        self, max_depth: int = 8, min_actions: int = 1
    ) -> tuple[Optional[SyntheticTree], Optional[Exception]]:
        """Wrapper for `self.generate()` to catch all errors."""
        try:
            st = self.generate(max_depth=max_depth, min_actions=min_actions)
        except (
            NoReactantAvailableError,
            NoBiReactionAvailableError,
            NoReactionAvailableError,
            NoReactionPossibleError,
            NoMergeReactionPossibleError,
            MaxNumberOfActionsError,
        ) as e:
            logger.error(e)
            return None, e

        except Exception as e:
            logger.error(e, exc_info=e, stack_info=False)
            return None, e
        else:
            return st, None


class SynTreeGeneratorPostProc:
    def __init__(self) -> None:
        pass

    @staticmethod
    def parse_generate_safe(
        results: List[Tuple[Union[SyntheticTree, None], Union[Exception, None]]]
    ) -> tuple[SyntheticTreeSet, dict[str, int]]:
        """Parses the result from `SynTreeGenerator.generate_safe`.
        In particular:
            - parses valid SynTrees and returns a `SyntheticTreeSet`
            - counts error messages and returns a `dict`
        """
        from collections import Counter

        if isinstance(results, tuple):
            results = [results]

        syntrees, exits = zip(*results)
        exit_codes = [
            e.__class__.__name__ if e is not None else "success" for e in exits
        ]
        valied_syntrees = [st for st in syntrees if st is not None]
        return SyntheticTreeSet(valied_syntrees), dict(Counter(exit_codes))


def load_syntreegenerator(file: str) -> SynTreeGenerator:
    import pickle

    with open(file, "rb") as f:
        syntreegenerator = pickle.load(f)
    return syntreegenerator


def save_syntreegenerator(syntreegenerator: SynTreeGenerator, file: str) -> None:
    import pickle

    with open(file, "wb") as f:
        pickle.dump(syntreegenerator, f)


# TODO: Move all these encoders to "from syn_net.encoding/"
# TODO: Evaluate if One-Hot-Encoder can be replaced with encoder from sklearn


ItemType = TypeVar("ItemType", bound=np.generic)


class OneHotEncoder:
    def __init__(self, d: int) -> None:
        self.d = d

    def __repr__(self) -> str:
        return f"'{self.__class__.__name__}': {self.__dict__}"

    @property
    def args(self) -> dict[str, Any]:
        return {**self.__dict__, **{"name": self.__class__.__name__}}

    @property
    def nbits(self) -> int:
        return self.get_nbits()

    def encode(self, ind: int, datatype: type = np.float64) -> npt.NDArray[ItemType]:
        """Returns a (1,d)-array with zeros and a 1 at index `ind`."""
        onehot: npt.NDArray[Any]
        onehot = np.zeros((1, self.d), dtype=datatype)  # (1,d)
        onehot[0, ind] = 1.0
        return onehot  # (1,d)

    def get_nbits(self) -> int:
        """Return the dimensionality as nbits."""
        return self.d


class MorganFingerprintEncoder:
    def __init__(self, radius: int, nbits: int) -> None:
        self.radius = radius
        self._nbits = nbits

    def __repr__(self) -> str:
        return f"'{self.__class__.__name__}': {self.__dict__}"

    @property
    def args(self) -> dict[str, Any]:
        return {**self.__dict__, **{"name": self.__class__.__name__}}

    @property
    def nbits(self) -> int:
        return self.get_nbits()

    def get_nbits(self) -> int:
        """Return the number of bits aka. the dimensionality.

        Properties and inheritance is not trivial and hence the property nbits maps to this function
        which is used for inheritance.
        """
        return self._nbits

    def encode(
        self, smi: Optional[str], allow_none: bool = True
    ) -> npt.NDArray[np.float_]:
        mol: Optional[Chem.Mol] = None
        if smi is not None:
            mol = Chem.MolFromSmiles(smi)

        # Cath these pesky Nones
        if mol is None:
            if allow_none:  # If allowed return a vector of zeros
                return np.zeros((1, self.nbits))  # (1,d)
            # else raise an error
            if smi is None:
                raise ValueError(f"SMILES cannot be None if `{allow_none=}`.")
            raise ValueError(f"SMILES ({smi}) encodes invalid molecule.")

        # Encode the molecule
        fp = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nbits)
        )
        return fp.reshape((1, -1))  # (1,d)

    def encode_batch(
        self, smis: Iterable[Optional[str]], allow_none: bool = True
    ) -> npt.NDArray[np.float_]:
        """Encode a batch.

        Info: Added for convenience for datasets to encode a state (target,root1,root2) in one go
        """
        return np.asarray(
            [self.encode(smi, allow_none) for smi in smis]
        ).squeeze()  # (num_items, nbits)


class IdentityIntEncoder:
    def __init__(self) -> None:
        pass

    @property
    def args(self) -> dict[str, Any]:
        return {**self.__dict__, **{"name": self.__class__.__name__}}

    def __repr__(self) -> str:
        return f"'{self.__class__.__name__}': {self.__dict__}"

    @property
    def nbits(self) -> int:
        return self.get_nbits()

    def encode(self, number: int) -> npt.NDArray[np.int_]:
        return np.atleast_2d(number)

    def get_nbits(self) -> int:
        return 1


class SynTreeFeaturizer:
    def __init__(
        self,
        *,
        reactant_embedder: MorganFingerprintEncoder,
        mol_embedder: MorganFingerprintEncoder,
        rxn_embedder: IdentityIntEncoder,
        action_embedder: IdentityIntEncoder,
    ) -> None:
        # Embedders
        self.reactant_embedder = reactant_embedder
        self.mol_embedder = mol_embedder
        self.rxn_embedder = rxn_embedder
        self.action_embedder = action_embedder

    def __repr__(self) -> str:
        return f"{self.__dict__}"

    def featurize(
        self, syntree: SyntheticTree
    ) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
        """Featurize a synthetic tree at every state.

        Note:
          - At each iteration of the syntree growth, an action is chosen
          - Every action (except "end") comes with a reaction.
          - For every action, we compute:
            - a "state"
            - a "step", a vector that encompasses all info we need for training the neural nets.
              This step is: [action, z_rt1, reaction_id, z_rt2, z_root_mol_1]
        """

        states: list[npt.NDArray[np.float_]] = []
        steps: list[npt.NDArray[np.float_]] = []
        if syntree.root is None:
            raise ValueError("Root is None.")
        target_mol = syntree.root.smiles
        z_target_mol = self.mol_embedder.encode(target_mol)

        # Recall: We can have at most 2 sub-trees, each with a root node.
        mol1: Optional[str] = None
        mol2: Optional[str] = None
        root_mol_1 = None
        root_mol_2 = None
        rxn_node: Optional[NodeRxn] = None
        for i, action in enumerate(syntree.actions):
            # 1. Encode "state"
            if root_mol_1 is None or root_mol_2 is None:
                raise ValueError("Root molecules are not set.")
            z_root_mol_1 = self.mol_embedder.encode(root_mol_1)
            z_root_mol_2 = self.mol_embedder.encode(root_mol_2)
            state = np.atleast_2d(
                np.concatenate((z_root_mol_1, z_root_mol_2, z_target_mol), axis=1)
            )  # (1,3d)

            # 2. Encode "super"-step
            if action == 3:  # end
                if rxn_node is None:
                    raise ValueError("Reaction node is None.")
                step = np.concatenate(
                    (
                        self.action_embedder.encode(action),
                        self.reactant_embedder.encode(mol1),
                        self.rxn_embedder.encode(rxn_node.rxn_id),
                        self.reactant_embedder.encode(mol2),
                        self.mol_embedder.encode(mol1),
                    ),
                    axis=1,
                )
            else:
                rxn_node = syntree.reactions[i]

                if len(rxn_node.child) == 1:
                    mol1 = rxn_node.child[0]
                    mol2 = None
                elif len(rxn_node.child) == 2:
                    mol1 = rxn_node.child[0]
                    mol2 = rxn_node.child[1]
                else:  # TODO: Change `child` is stored in reaction node so we can just unpack via *
                    raise ValueError()

                step = np.concatenate(
                    (
                        self.action_embedder.encode(action),
                        self.reactant_embedder.encode(mol1),
                        self.rxn_embedder.encode(rxn_node.rxn_id),
                        self.reactant_embedder.encode(mol2),
                        self.mol_embedder.encode(mol1),
                    ),
                    axis=1,
                )

            # 3. Prepare next iteration
            if action == 2:  # merge
                root_mol_1 = rxn_node.parent
                root_mol_2 = None

            elif action == 1:  # expand
                root_mol_1 = rxn_node.parent

            elif action == 0:  # add
                root_mol_2 = root_mol_1
                root_mol_1 = rxn_node.parent

            # 4. Keep track of data
            states.append(state)
            steps.append(step)

        # Some housekeeping on dimensions
        states_matrix = np.atleast_2d(np.asarray(states).squeeze())
        steps_matrix = np.atleast_2d(np.asarray(steps).squeeze())

        return sparse.csc_matrix(states_matrix), sparse.csc_matrix(steps_matrix)
