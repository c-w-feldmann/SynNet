"""Decoder for a molecular embedding."""
# Setup
# * Imports
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import lightning as ltg
import torch

from synnet.data_generation.preprocessing import Reaction
from synnet.data_generation.syntrees import MorganFingerprintEncoder, OneHotEncoder

from sklearn.metrics.pairwise import cosine_distances

from synnet.utils.data_utils import (
    ReactionSet,
    SyntheticTree,
    SyntheticTreeSet,
)
from synnet.utils.synnet_exceptions import NoSuitableReactantError, StateEmbeddingError
from synnet.utils.custom_types import PathType
from synnet.encoding.embedding import MolecularEmbeddingManager

# * Logging
logger = logging.getLogger(__name__)

# Class definitions


class HelperDataloader:
    @classmethod
    def _fetch_data_chembl(cls, file: PathType) -> list[str]:
        df = pd.read_csv(file, sep="\t")
        smis_query = df["smiles"].to_list()
        return smis_query

    @classmethod
    def _fetch_data_from_file(cls, file: PathType) -> list[str]:
        with open(file, "rt") as f:
            smis_query = [line.strip() for line in f]
        return smis_query

    @classmethod
    def fetch_data(cls, file_str: PathType) -> list[str]:
        file_path = Path(file_str)
        if any(split in file_path.stem for split in ["train", "valid", "test"]):
            logger.info(f"Reading data from {file_path}")
            syntree_collection = SyntheticTreeSet().load(file_path)
            all_root_list = [syntree.root for syntree in syntree_collection.synthetic_tree_list]
            if None in all_root_list:
                raise ValueError("None in root_list")
            root_list = [root for root in all_root_list if root is not None]
            smiles = [root.smiles for root in root_list]
        elif "chembl" in file_path.stem:
            smiles = cls._fetch_data_chembl(file_path)
        else:  # Hopefully got a filename instead
            smiles = cls._fetch_data_from_file(file_path)
        return smiles


class SynTreeDecoder:
    """Decoder for a molecular embedding."""

    mol_encoder: MorganFingerprintEncoder
    rxn_collection: ReactionSet
    similarity_fct: Optional[Callable[[npt.NDArray[np.float_], List[str]], npt.NDArray[np.float_]]]

    def __init__(
        self,
        *,
        building_blocks_embedding_manager: MolecularEmbeddingManager,
        reaction_collection: ReactionSet,
        action_net: ltg.LightningModule,
        reactant1_net: ltg.LightningModule,
        rxn_net: ltg.LightningModule,
        reactant2_net: ltg.LightningModule,
        rxn_encoder: OneHotEncoder = OneHotEncoder(91),
        mol_encoder: MorganFingerprintEncoder = MorganFingerprintEncoder(2, 4096),
        similarity_fct: Optional[
            Callable[[npt.NDArray[np.float_], List[str]], npt.NDArray[np.float_]]
        ] = None,
    ) -> None:
        """Initialize a SynTreeDecoder.

        Parameters
        ----------
        building_blocks_embedding_manager : MolecularEmbeddingManager
            Embedding manager for the building blocks.
        reaction_collection : ReactionSet
            Collection of reactions.
        action_net : ltg.LightningModule
            Network for predicting the action.
        reactant1_net : ltg.LightningModule
            Network for predicting the first reactant.
        rxn_net : ltg.LightningModule
            Network for predicting the reaction.
        reactant2_net : ltg.LightningModule
            Network for predicting the second reactant.
        rxn_encoder : Encoder, optional
            Encoder for the reaction, by default OneHotEncoder(91)
        mol_encoder : MorganFingerprintEncoder
            Object for encoding molecules as vector, by default MorganFingerprintEncoder(2, 4096)
        similarity_fct : Optional[Callable[[npt.NDArray[np.float_], List[str]], npt.NDArray[np.float_]]], optional
            Similarity function for the reactants, by default None

        Returns
        -------
        None
        """
        # Assets
        self.bblocks_manager = building_blocks_embedding_manager
        self.rxn_collection = reaction_collection
        self.num_reactions = len(self.rxn_collection)

        # Encoders
        self.rxn_encoder = rxn_encoder
        self.mol_encoder = mol_encoder

        # Networks
        self.nets: Dict[str, ltg.LightningModule] = {
            "act": action_net,
            "rt1": reactant1_net,
            "rxn": rxn_net,
            "rt2": reactant2_net,
        }

        # Similarity fct
        self.similarity_fct = similarity_fct
        self.action_mapping = {
            0: "add",
            1: "expand",
            2: "merge",
            3: "end",
        }

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> Self:
        """Instantiate a SynTreeDecoder from a config dict."""
        return cls(**config)

    def _get_syntree_state_embedding(
        self, state: tuple[Optional[str], Optional[str]]
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Compute state embedding for a state.

        Parameters
        ----------
        state : tuple[Optional[str], Optional[str]]
            State of the syntree.
                First string is the root of the syntree.
                Second string is the root of the second syntree, if any.

        Returns
        -------
        tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
            State embedding.
        """
        nbits: int = self.mol_encoder.nbits
        if state[0] is None and state[1] is None:
            z_mol_root1 = np.zeros(nbits)
            z_mol_root2 = np.zeros(nbits)
        elif state[0] is not None and state[1] is None:  # Only one actively growing syntree
            z_mol_root1 = self.mol_encoder.encode(state[0])
            z_mol_root2 = np.zeros(nbits)
        elif state[0] is not None and state[1] is not None:  # Two syntrees
            z_mol_root1 = self.mol_encoder.encode(state[0])
            z_mol_root2 = self.mol_encoder.encode(state[1])
        else:
            raise StateEmbeddingError(f"Unable to compute state embedding. Passed {state=}")

        return np.squeeze(z_mol_root1), np.squeeze(z_mol_root2)

    def get_state_embedding(
        self, state: tuple[Optional[str], Optional[str]], z_target: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Computes embeddings for all molecules in the input space.

        Embedding = [z_mol1, z_mol2, z_target]

        TODO: Handle None None

        Parameters
        ----------
        state : tuple[str, Optional[str]]
            State of the syntree.
        z_target : npt.NDArray[np.float_]
            Embedding of the target molecule.

        Returns
        -------
        npt.NDArray[np.float_]
            Embedding of the state.
        """
        z_mol_root1, z_mol_root2 = self._get_syntree_state_embedding(state)
        z_state = np.concatenate([z_mol_root1, z_mol_root2, z_target], axis=0)
        return z_state  # (d,)

    def _get_action_mask(self, syntree: SyntheticTree) -> npt.NDArray[np.bool_]:
        """Get a mask of possible action for a SyntheticTree"""
        # Recall: (Add, Expand, Merge, and End)
        can_add = False
        can_merge = False
        can_expand = False
        can_end = False

        state = syntree.get_state()
        if state[0] is None and state[1] is None:  # base case
            can_add = True
        elif state[0] is not None and state[1] is None and (syntree.depth == self.max_depth - 1):
            # syntree is 1 update apart from its max depth, only allow to end it.
            can_end = True
        elif state[0] is not None and state[1] is None:
            can_add = True
            can_expand = True
            can_end = True
        elif state[0] is not None and state[1] is not None:
            can_expand = True  # TODO: do not expand when we're 2 steps away from max depth
            can_merge = any(self.get_reaction_mask((state[0], state[1])))
        else:
            raise ValueError(f"Invalid state: {state}")

        return np.array((can_add, can_expand, can_merge, can_end), dtype=bool)

    def _find_valid_bimolecular_rxns(
        self, reactants: Tuple[Optional[str], Optional[str]]
    ) -> npt.NDArray[np.bool_]:
        reaction_mask: list[bool] = []
        for _rxn in self.rxn_collection.rxns:
            try:
                p = _rxn.run_reaction(reactants, allow_to_fail=False)
                is_valid_reaction = p is not None
            except Exception as e:
                #print(e)  # TODO: implement reaction.can_react(reactants) method returning a bool
                # run_reactions() does some validity-checks and raises Exception
                is_valid_reaction = False
            reaction_mask += [is_valid_reaction]
        return np.asarray(reaction_mask)

    def _find_valid_unimolecular_rxns(self, reactant: str) -> npt.NDArray[np.bool_]:
        reaction_mask: list[bool]
        reaction_mask = [rxn.is_reactant(reactant) for rxn in self.rxn_collection.rxns]
        final_mask = []

        # Remove reactions without available reactants as partner
        for _rxn, is_selected in zip(
            self.rxn_collection.rxns, reaction_mask
        ):  # type: Reaction, bool
            if not is_selected:
                final_mask.append(False)
            elif _rxn.num_reactant == 1:
                final_mask.append(is_selected)
            elif _rxn.is_reactant_first(reactant):
                if _rxn.available_reactants is None:
                    final_mask.append(False)
                elif len(_rxn.available_reactants[1]) == 0:
                    final_mask.append(False)
                else:
                    final_mask.append(is_selected)
            elif _rxn.is_reactant_second(reactant):
                if _rxn.available_reactants is None:
                    final_mask.append(False)
                elif len(_rxn.available_reactants[0]) == 0:
                    final_mask.append(False)
                else:
                    final_mask.append(is_selected)
        return np.asarray(final_mask)

    def get_reaction_mask(self, reactants: Union[str, tuple[str, str]]) -> npt.NDArray[np.bool_]:
        """Get a mask of possible reactions for given reactants.

        Parameters
        ----------
        reactants : Union[str, tuple[str, str]]
            Reactant required for reaction.

        Returns
        -------
        npt.NDArray[np.bool_]
            Mask of possible reactions.
        """
        if isinstance(reactants, str):
            return self._find_valid_unimolecular_rxns(reactants)
        else:
            return self._find_valid_bimolecular_rxns(reactants)

    def decode(
        self,
        z_target: npt.NDArray[np.float_],
        *,
        k_reactant1: int = 1,
        max_depth: int = 15,
        debug: bool = False,
        **kwargs: Any,
    ) -> tuple[SyntheticTree, Optional[float]]:
        """Decode a target molecule into a SyntheticTree.

        TODO: This needs to be refactored and rewritten to smaller methods.

        Parameters
        ----------
        z_target : npt.NDArray[np.float_]
            Target molecule embedding.
        k_reactant1 : int, optional
            Number of reactants to sample for the first reactant, by default 1
        max_depth : int, optional
            Maximum depth of the synthetic tree, by default 15
        debug : bool
            Activate debug mode
        **kwargs: Any
            Additional arguments to pass to the decoder.

        Returns
        -------
        SyntheticTree
            SyntheticTrees
        Optional[float]
            Similarity of the target molecule and the final molecule.
        """
        if debug:
            logger.setLevel("DEBUG")

        # small constant is added to probabilities so that masked values (=0.0) are unequal to probabilites (0+eps)
        eps = 1e-6

        self.max_depth = max_depth  # so we can access this param in methods
        act, rt1, rxn, rt2 = self.nets.values()
        z_target = np.squeeze(z_target)  # TODO: Fix shapes
        syntree = SyntheticTree()
        mol_recent: Optional[str] = None  # most-recent root mol
        i = 0
        while syntree.depth < self.max_depth:
            logger.debug(f"Iteration {i} | {syntree.depth=}")

            # Current state
            state = syntree.get_state()
            z_state = self.get_state_embedding(state, z_target)  # (3d)
            z_state_tensor = torch.Tensor(z_state[None, :])  # (1,3d)
            z_state_tensor  = z_state_tensor.to(act.device)

            # Prediction action
            p_action = act.forward(z_state_tensor)  # (1,4)
            p_action = p_action.detach().cpu().numpy() + eps
            action_mask = self._get_action_mask(syntree)
            action_id = int(np.argmax(p_action * action_mask))
            logger.debug(f" Action: {self.action_mapping[action_id]}. ({p_action.round(2)=})")

            if self.action_mapping[action_id] == "end":
                break
            elif self.action_mapping[action_id] == "add":
                # Start a new sub-syntree.
                # TODO: z=z' as mol embedding dim is differnt
                z_reactant1 = rt1.forward(z_state_tensor)
                z_reactant1 = z_reactant1.detach().cpu().numpy()  # (1,d')

                # Select building block via kNN search
                k = k_reactant1 if i == 0 else 1
                logger.debug(f"  k-NN search for 1st reactant with k={k}.")
                idxs = self.bblocks_manager.kdtree.query(z_reactant1, k=k, return_distance=False)
                # idxs.shape = (1,k)
                idx = idxs[0][k - 1]
                reactant_1: Optional[str] = self.bblocks_manager.smiles_array[idx]
                logger.debug(f"  Selected 1st reactant ({idx=}): `{reactant_1}`")
            elif self.action_mapping[action_id] == "expand":
                # We already have a 1st reactant.
                reactant_1 = mol_recent  # aka root mol (=product) from last iteration
            elif self.action_mapping[action_id] == "merge":
                # We already have 1st + 2nd reactant
                # TODO: If we merge two trees, we have to determine a reaction.
                #       This means we have to encode the "1st reactant"
                #       -> Investitage if this is
                #           - the `mol_recent`
                #           - the root mol of the "other" syntree
                #       Note: It does not matter for the reaction, but it does
                #             matter for the reaction prediction.
                reactant_1 = mol_recent
            else:
                raise ValueError(f"Unknown action id: {action_id}")

            if reactant_1 is None:
                raise AssertionError("No reactant_1 found.")
            # Predict reaction
            z_reactant_1 = torch.Tensor(self.mol_encoder.encode(reactant_1)).to(z_state_tensor.device)  # (1,d)
            x = torch.cat((z_state_tensor, z_reactant_1), dim=1)
            # x = torch.Tensor(np.concatenate((z_state_tensor, z_reactant_1), axis=1))
            p_rxn = rxn.forward(x)
            p_rxn = p_rxn.detach().cpu().numpy() + eps
            logger.debug(
                "  Top 5 reactions: "
                + ", ".join(
                    [
                        f"{__idx:>2d} (p={p_rxn[0][__idx]:.2f})"
                        for __idx in np.argsort(p_rxn)[0, -5:][::-1]
                    ]
                )
            )

            # Reaction mask
            if self.action_mapping[action_id] == "merge":
                reactant_2 = (set(state) - {reactant_1}).pop()
                # TODO: fix these shenanigans and determine reliable which is the 2nd reactant
                #       by "knowing" the order of the state
                # if merge, only allow bi-mol rxn
                if reactant_2 is None:
                    raise AssertionError("Merge action requires two reactants.")
                reaction_mask = self.get_reaction_mask((reactant_1, reactant_2))
            else:  # add or expand (both start from 1 reactant only)
                reaction_mask = self.get_reaction_mask(reactant_1)
            logger.debug(f"  Reaction mask with n choices: {reaction_mask.sum()}")

            # Check: Are any reactions possible? If not, break. # TODO: Cleanup
            if not any(reaction_mask):
                # Is it possible to gracefully end this tree?
                # If there is only a sinlge tree, mark it as "ended"
                if len(state) == 1:
                    action_id = 3
                logger.debug(
                    f"Terminated decoding as no reaction is possible, manually enforced {action_id=} "
                )
                break

            # Select reaction template
            rxn_id: int = int(np.argmax(p_rxn * reaction_mask))
            reaction: Reaction = self.rxn_collection.rxns[rxn_id]
            logger.debug(
                f"  Selected {'bi' if reaction.num_reactant==2 else 'uni'} reaction {rxn_id=}"
            )
            if reaction.available_reactants is None:
                raise AssertionError(f"Reaction has no available reactants: {reaction.smirks}")
            # We have three options:
            #  1. "merge" -> need to sample 2nd reactant
            #  2. "expand" or "expand" -> only sample 2nd reactant if reaction is bimol
            reactant_2 = None
            if self.action_mapping[action_id] == "merge":
                reactant_2 = syntree.get_state()[1]  # "old" root mol, i.e. other in state
            elif self.action_mapping[action_id] in ["add", "expand"]:
                if reaction.num_reactant == 2:
                    # Sample 2nd reactant
                    z_rxn = torch.Tensor(self.rxn_encoder.encode(rxn_id)).to(z_state_tensor.device)
                    x = torch.cat([z_state_tensor, z_reactant_1, z_rxn], dim=1)

                    z_reactant2 = rt2.forward(x)
                    z_reactant2 = z_reactant2.cpu().detach().numpy()

                    # Select building block via kNN search
                    # does reactant 1 match position 0 or 1 in the template?
                    if reaction.is_reactant_first(reactant_1):
                        # TODO: can match pos 1 AND 2 at teh same time
                        available_reactants_2 = reaction.available_reactants[1]
                    else:
                        available_reactants_2 = reaction.available_reactants[0]

                    # Get smiles -> index -> embedding
                    # Get position of available reactants in bblocks_manager
                    _emb = self.bblocks_manager.get_embedding_for(available_reactants_2)
                    if _emb.shape[0] == 0:
                        raise NoSuitableReactantError(
                            f"No reactant found for reaction: {reaction.smirks}"
                        )
                    logger.debug(f"  Subspace of available 2nd reactants: {_emb.shape[0]} ")
                    _dists = cosine_distances(_emb, z_reactant2)
                    idx = np.argmin(_dists)
                    reactant_2 = available_reactants_2[idx]
                    logger.debug(f"  Selected 2nd reactant ({idx=}): `{reactant_2}`")
                else:  # this is a unimolecular reaction
                    reactant_2 = None

            # Run reaction
            product = reaction.run_reaction((reactant_1, reactant_2), allow_to_fail=False)
            logger.debug(f"  Ran reaction {reactant_1} + {reactant_2} -> {product}")

            # Validate outcome of reaction
            if product is None:
                error_msg = (
                    "rdkit.RunReactants() produced invalid product. "
                    + f"Reaction ID: {rxn_id}, {syntree.depth=} "
                    + f"Reaction: `{reactant_1} + {reactant_2} -> {product}`"
                )
                logger.error("  " + error_msg)
                # Is it possible to gracefully end this tree?
                # If there is only a sinlge tree, mark it as "ended"
                if len(state) == 1:
                    action_id = 3
                logger.debug(
                    f"Terminated decoding as no reaction is possible, manually enforced {action_id=} "
                )
                break

            # Update
            logger.debug("  Updating SynTree.")
            syntree.update(int(action_id), int(rxn_id), reactant_1, reactant_2, product)
            mol_recent = product
            i += 1

        # End of generation. Validate outcome
        if self.action_mapping[action_id] == "end":
            syntree.update(int(action_id), None, None, None, None)

        # Compute similarity to target
        if syntree.is_valid and self.similarity_fct is not None:
            similarities = float(
                np.max(
                    self.compute_similarity_to_target(
                        similarity_fct=self.similarity_fct,
                        z_target=z_target,
                        syntree=syntree,
                    )
                )
            )
        else:
            similarities = None

        return syntree, similarities

    def compute_similarity_to_target(
        self,
        *,
        similarity_fct: Callable[[npt.NDArray[np.float_], list[str]], npt.NDArray[np.float_]],
        z_target: npt.NDArray[np.float_],
        syntree: SyntheticTree,
    ) -> npt.NDArray[np.float_]:  # TODO: move to its own class?
        """Computes the similarity to a `z_target` for all nodes, as
        we can in theory truncate the tree to our liking.
        """
        return np.array(similarity_fct(z_target, [smi for smi in syntree.nodes_as_smiles]))


class SynTreeDecoderGreedy:
    def __init__(self, decoder: SynTreeDecoder) -> None:
        self.decoder = decoder  # composition over inheritance

    def decode(
        self,
        z_target: npt.NDArray[np.float_],
        *,
        attempts: int = 3,
        objective: Optional[str] = "best",  # "best", "best+shortest"
        debug: bool = False,
    ) -> tuple[SyntheticTree, Optional[float]]:
        """Decode `z_target` at most `attempts`-times and return the most-similar one."""

        best_similarity = -np.inf
        best_syntree = SyntheticTree()
        i = 0
        for i in range(attempts):
            logger.debug(f"Greedy search attempt: {i} (k_reactant1={i+1})")

            syntree, max_similarity = self.decoder.decode(
                z_target,
                k_reactant1=i + 1,
                debug=debug,
            )

            #  ↓ for legacy decoder, which could return None
            if syntree is None or not syntree.is_valid:
                continue
            # Sanity check:
            if not max_similarity:
                raise ValueError("Did you specify a `similarity_fct` for the decoder?")

            # Do we have a new best candidate?
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_syntree = syntree
            logger.debug(f"  Max similarity: {max_similarity:.3f} (best: {best_similarity:.3f})")
            if objective == "best" and best_similarity == 1.0:
                logger.debug(
                    f"Decoded syntree has similarity 1.0 and {objective=}; abort greedy search."
                )
                break

        # Return best results
        return best_syntree, best_similarity
