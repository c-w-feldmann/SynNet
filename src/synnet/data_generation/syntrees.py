"""syntrees
"""
import logging
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from scipy import sparse
from tqdm import tqdm

from synnet.config import MAX_PROCESSES

logger = logging.getLogger(__name__)

from synnet.data_generation.exceptions import (
    MaxNumberOfActionsError,
    NoBiReactionAvailableError,
    NoMergeReactionPossibleError,
    NoReactantAvailableError,
    NoReactionAvailableError,
    NoReactionPossibleError,
)
from synnet.utils.data_utils import Reaction, ReactionSet, SyntheticTree, SyntheticTreeSet


class SynTreeGenerator:
    """Generates synthetic trees by randomly applying reactions to building blocks."""

    ACTIONS: dict[int, str] = {i: action for i, action in enumerate("add expand merge end".split())}

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        rxn_collection: Optional[ReactionSet] = None,
        rng=np.random.default_rng(),  # Note: When generating syntrees with mp provide a fresh seed!
        processes: int = MAX_PROCESSES,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates
        self.rxns = [Reaction(template=tmplt) for tmplt in rxn_templates]
        self.rng = rng
        self.IDX_RXNS = np.arange(len(self.rxns))
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
            self.rxns = rxn_collection
            self.rxns_initialised = True

    def __match_mp(self):
        # TODO: refactor / merge with `BuildingBlockFilter`
        # TODO: Rename `ReactionSet` -> `ReactionCollection` (same for `SyntheticTreeSet`)
        from functools import partial

        from pathos import multiprocessing as mp

        def __match(_rxn: Reaction, *, bblocks: list[str]):
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, bblocks=self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            rxns = pool.map(func, self.rxns)

        self.rxns = rxns
        return self

    def _init_rxns_with_reactants(self):
        """Initializes a `Reaction` with a list of possible reactants.

        Info: This can take a while for lots of possible reactants."""
        if self.processes == 1:
            rxns = tqdm(self.rxns) if self.verbose else self.rxns
            self.rxns = [rxn.set_available_reactants(self.building_blocks) for rxn in rxns]
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

    def _find_rxn_candidates(self, smiles: str, raise_exc: bool = True) -> list[bool]:
        """Find reactions which reactions have `smiles` as reactant."""
        rxn_mask = [rxn.is_reactant(smiles) for rxn in self.rxns]

        if raise_exc and not any(rxn_mask):  # Do not raise exc when checking if two mols can react
            raise NoReactionAvailableError(f"Cannot find a reaction for reactant: {smiles}.")
        return rxn_mask

    def _sample_rxn(self, mask: Optional[np.ndarray] = None) -> Tuple[Reaction, int]:
        """Sample a reaction by index."""
        if mask is None:
            irxn_mask = self.IDX_RXNS  #  = all reactions are possible
        else:
            mask = np.asarray(mask)  # If a mask is passed, it should have at least 1 True entry
            irxn_mask = self.IDX_RXNS[mask]

        idx = self.rng.choice(irxn_mask)
        rxn = self.rxns[idx]

        logger.debug(
            f"    Sampled {'uni' if rxn.num_reactant == 1 else 'bi'}-molecular reaction with id {idx:d}"
        )
        return rxn, idx

    def _expand(
        self, reactant_1: str, raise_exc: bool = True
    ) -> Tuple[str, Optional[str], Optional[str], np.int64]:
        """Expand a sub-tree from one molecule.
        This can result in uni- or bimolecular reaction."""

        # Identify applicable reactions
        rxn_mask = self._find_rxn_candidates(reactant_1)

        # Sample reaction (by index)
        rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)

        # Sample 2nd reactant
        if rxn.num_reactant == 1:
            reactant_2 = None
        else:
            # Sample a molecule from the available reactants of this reaction
            # That is, for a reaction A + B -> C,
            #   - determine if we have "A" or "B"
            #   - then sample "B" (or "A")
            reactant_2_order = 1 if rxn.is_reactant_first(reactant_1) else 0
            available_reactants = rxn.available_reactants[reactant_2_order]

            if (nAvailableReactants := len(available_reactants)) == 0:
                raise NoReactantAvailableError(
                    f"No reactant available for bimolecular reaction (ID: {idx_rxn}). Present reactant: {reactant_1}. Missing {'second' if reactant_2_order==1 else 'first'} reactant."
                )

            idx = self.rng.choice(nAvailableReactants)
            reactant_2 = available_reactants[idx]
            logger.debug(f"    Sampled second reactant: {reactant_2}")

        # Run reaction
        reactants = (reactant_1, reactant_2)
        product = rxn.run_reaction(reactants)
        if raise_exc and product is None:
            raise NoReactionPossibleError(
                f"Reaction (ID: {idx_rxn}) not possible with: `{reactant_1} + {reactant_2}`."
            )
        return *reactants, product, idx_rxn

    def _merge(self, syntree: SyntheticTree):
        """Merge the two root mols in the `SyntheticTree`"""
        # Identify suitable rxn
        r1, r2 = syntree.get_state()
        rxn_mask = self._get_rxn_mask(tuple((r1, r2)))
        # Sample reaction
        rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)
        # Run reaction
        p = rxn.run_reaction((r1, r2))
        if p is None:
            raise NoMergeReactionPossibleError(
                f"Reaction (ID: {idx_rxn}) not possible with: {r1} + {r2}."
            )
        return r1, r2, p, idx_rxn

    def _get_action_mask(self, syntree: SyntheticTree):
        """Get a mask of possible action for a SyntheticTree"""
        # Recall: (Add, Expand, Merge, and End)
        canAdd = False
        canMerge = False
        canExpand = False
        canEnd = False

        state = syntree.get_state()
        nTrees = len(state)
        if nTrees == 0:  # base case
            canAdd = True
        elif nTrees == 1:
            canAdd = True
            canExpand = True
            canEnd = True
        elif nTrees == 2:
            canExpand = True
            canMerge = any(self._get_rxn_mask(tuple(state), raise_exc=False))
        else:
            raise ValueError

        # Special cases.
        # Case: Syntree is 1 update apart from its max size, only allow to end it.
        if nTrees == 1 and (syntree.num_actions == self.max_depth - 1):
            logger.debug(
                f"  Overriding action space to only allow action=end."
                + f"({nTrees=}, {syntree.num_actions=}, {self.max_depth=})"
            )
            canAdd, canMerge, canExpand = False, False, False
            canEnd = True
        elif nTrees == 2 and (syntree.num_actions == self.max_depth - 2):
            # ATTN: This might result in an Exception,
            #       i.e. when no rxn template matches or the product is invalid etc.
            logger.debug(
                f"  Overriding action space to forcefully merge trees."
                + f"({nTrees=}, {syntree.num_actions=}, {self.max_depth=})"
            )
            canAdd, canExpand, canEnd = False, False, False
            canMerge = True
        elif nTrees == 1 and (syntree.num_actions == self.max_depth - 3):
            # Handle case for max_depth=6 and [0, 1, 1, 1, 0, 2, 3] ?
            #                                              ^-- prevent this
            pass
        # Case: Syntree is 2 updates away from its max size, only allow to merge it.
        if syntree.num_actions < self.min_actions:
            canEnd = False

        return np.array((canAdd, canExpand, canMerge, canEnd), dtype=bool)

    def _get_rxn_mask(self, reactants: tuple[str, str], raise_exc=True) -> list[bool]:
        """Get a mask of possible reactions for the two reactants."""
        # First: Identify bi-molecular reactions
        masks_bimol = [rxn.num_reactant == 2 for rxn in self.rxns]  # TODO: Cache?
        # Second: Check if reactants match template in correct or reversed order, i.e.
        #         check if (r1->position1 & r2->position2) "ordered"
        #         or       (r1->position2 & r2->position1) "reversed"
        r1, r2 = reactants
        masks_r1 = [
            (rxn.is_reactant_first(r1), rxn.is_reactant_second(r1)) if is_bi else (False, False)
            for is_bi, rxn in zip(masks_bimol, self.rxns)
        ]
        masks_r2 = [
            (rxn.is_reactant_first(r2), rxn.is_reactant_second(r2)) if is_bi else (False, False)
            for is_bi, rxn in zip(masks_bimol, self.rxns)
        ]

        # Check if reactants match template the ordered or reversed way
        arr = np.array((masks_r1, masks_r2))  # (nReactant, nReaction, first-second-position)
        arr = arr.swapaxes(0, 1)  # view:         (nReaction, nReactant, first-second-position)
        canReactOrdered = np.trace(arr, axis1=1, axis2=2) > 1  # (nReaction,)
        canReactReversed = np.flip(arr, axis=1).trace(axis1=1, axis2=2) > 1  # (nReaction,)
        mask = np.logical_or(canReactOrdered, canReactReversed).tolist()

        if raise_exc and not any(mask):
            raise NoBiReactionAvailableError(f"No reaction available for {reactants}.")
        return mask

    def _sample_action(self, syntree: SyntheticTree) -> int:
        """Samples an action conditioned on the state of the `SyntheticTree`"""
        p_action = self.rng.random((1, 4))  # (1,4)
        action_mask = self._get_action_mask(syntree)  # (1,4)
        act = np.argmax(p_action * action_mask)  # (1,)

        logger.debug(f"  Sampled action: {self.ACTIONS[act]}")
        return act

    def generate(self, max_depth: int = 8, min_actions: int = 1):
        """Generate a syntree by random sampling."""
        assert min_actions < max_depth, "min_actions must be smaller than max_depth."
        assert max_depth > 1, "max_actions must be larger than 1. (smallest treee is [`add`,`end`]"

        logger.debug(f"Starting synthetic tree generation with {min_actions=} and {max_depth=} ")
        self.max_depth = max_depth  # TODO: rename to reflect "number of actions"
        self.min_actions = min_actions

        # Init
        syntree = SyntheticTree()

        for i in range(max_depth + 1):
            logger.debug(f"Iter {i} | {syntree.depth=} | num_actions={syntree.actions.__len__()} ")

            # Sample action
            act = self._sample_action(syntree)
            action = self.ACTIONS[act]

            if action == "end":
                r1, r2, p, idx_rxn = None, None, None, -1
            elif action == "expand":
                # Expand this subtree: reaction, (reactant2), run it.
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
            syntree.update(action=int(act), rxn_id=int(idx_rxn), mol1=r1, mol2=r2, mol_product=p)
            logger.debug(f"SynTree updated.")
            if action == "end":
                break

        if syntree.num_actions > max_depth and not action == "end":
            raise MaxNumberOfActionsError(
                f"Maximum number of actions exceeded. ({syntree.actions=}>{max_depth})."
            )
        logger.debug(f"🙌 SynTree completed.")
        return syntree

    def generate_safe(
        self, *args, max_depth: int = 8, min_actions: int = 1
    ) -> Tuple[Union[SyntheticTree, None], Union[Exception, None]]:
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
    ):
        """Parses the result from `SynTreeGenerator.generate_safe`.
        In particular:
            - parses valid SynTrees and returns a `SyntheticTreeSet`
            - counts error messages and returns a `dict`
        """
        from collections import Counter

        if isinstance(results, tuple):
            results = [results]

        syntrees, exits = zip(*results)
        exit_codes = [e.__class__.__name__ if e is not None else "success" for e in exits]
        syntrees = [st for st in syntrees if st is not None]
        return SyntheticTreeSet(syntrees), dict(Counter(exit_codes))


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

from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def encode(self, *args, **kwargs):
        ...

    def __repr__(self) -> str:
        return f"'{self.__class__.__name__}': {self.__dict__}"

    @property
    def args(self) -> dict:
        return {**self.__dict__, **{"name": self.__class__.__name__}}


class OneHotEncoder(Encoder):
    def __init__(self, d: int) -> None:
        self.d = d

    def encode(self, ind: int, datatype: np.dtype = np.float64) -> np.ndarray:
        """Returns a (1,d)-array with zeros and a 1 at index `ind`."""
        onehot = np.zeros((1, self.d), dtype=datatype)  # (1,d)
        onehot[0, ind] = 1.0
        return onehot  # (1,d)


class MorganFingerprintEncoder(Encoder):
    def __init__(self, radius: int, nbits: int) -> None:
        self.radius = radius
        self.nbits = nbits

    def encode(self, smi: str, allow_none: bool = True) -> np.ndarray:
        if not allow_none and smi is None:
            raise ValueError(f"SMILES ({smi=}) cannot be None if `{allow_none=}`.")
        if smi is None:
            fp = np.zeros((1, self.nbits))  # (1,d)
        else:
            mol = Chem.MolFromSmiles(smi)  # TODO: sanity check mol here or use datmol?
            bv = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nbits)
            fp = np.empty(self.nbits)
            Chem.DataStructs.ConvertToNumpyArray(bv, fp)
            fp = fp[None, :]
        return fp  # (1,d)

    def encode_batch(self, smis: Iterable[Union[str, None]], **kwargs) -> np.ndarray:
        """Encode a batch.

        Info: Added for convenience for datasets to encode a state (target,root1,root2) in one go"""
        return np.asarray(
            [self.encode(smi, **kwargs) for smi in smis]
        ).squeeze()  # (num_items, nbits)


class IdentityIntEncoder(Encoder):
    def __init__(self) -> None:
        pass

    def encode(self, number: int):
        return np.atleast_2d(number)


class SynTreeFeaturizer:
    def __init__(
        self,
        *,
        reactant_embedder: Encoder,
        mol_embedder: Encoder,
        rxn_embedder: Encoder,
        action_embedder: Encoder,
    ) -> None:
        # Embedders
        self.reactant_embedder = reactant_embedder
        self.mol_embedder = mol_embedder
        self.rxn_embedder = rxn_embedder
        self.action_embedder = action_embedder

    def __repr__(self) -> str:
        return f"{self.__dict__}"

    def featurize(self, syntree: SyntheticTree):
        """Featurize a synthetic tree at every state.

        Note:
          - At each iteration of the syntree growth, an action is chosen
          - Every action (except "end") comes with a reaction.
          - For every action, we compute:
            - a "state"
            - a "step", a vector that encompasses all info we need for training the neural nets.
              This step is: [action, z_rt1, reaction_id, z_rt2, z_root_mol_1]
        """

        states, steps = [], []

        target_mol = syntree.root.smiles
        z_target_mol = self.mol_embedder.encode(target_mol)

        # Recall: We can have at most 2 sub-trees, each with a root node.
        root_mol_1 = None
        root_mol_2 = None
        for i, action in enumerate(syntree.actions):

            # 1. Encode "state"
            z_root_mol_1 = self.mol_embedder.encode(root_mol_1)
            z_root_mol_2 = self.mol_embedder.encode(root_mol_2)
            state = np.atleast_2d(
                np.concatenate((z_root_mol_1, z_root_mol_2, z_target_mol), axis=1)
            )  # (1,3d)

            # 2. Encode "super"-step
            if action == 3:  # end
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
        states = np.atleast_2d(np.asarray(states).squeeze())
        steps = np.atleast_2d(np.asarray(steps).squeeze())

        return sparse.csc_matrix(states), sparse.csc_matrix(steps)
