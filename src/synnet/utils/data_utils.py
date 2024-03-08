"""
Here we define the following classes for working with synthetic tree data:
* `Reaction`
* `ReactionSet`
* `NodeChemical`
* `NodeRxn`
* `SyntheticTree`
* `SyntheticTreeSet`
"""

from __future__ import annotations

import logging

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import functools
import gzip
import itertools
import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from tqdm import tqdm

from synnet.utils.custom_types import PathType
from synnet.utils.synnet_exceptions import FailedReconstructionError


# the definition of reaction classes below
class Reaction:
    """A chemical reaction defined by a SMARTS pattern."""

    smirks: str
    reactant_template: tuple[str, ...]
    product_template: str
    agent_template: str
    available_reactants: Optional[
        tuple[list[str], ...]
    ]  # Cached list of available reactants for each template
    rxnname: str
    reference: Any
    weight: float

    def __init__(
        self,
        template: str,
        name: Optional[str] = None,
        reference: Optional[Any] = None,
        weight: float = 1.0,
        available_reactants: Optional[tuple[list[str], ...]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `Reaction`.

        Args:
            template: SMARTS string representing a chemical reaction.
            name: The name of the reaction for downstream analysis.
            reference: (placeholder)
        """
        self.smirks = template.strip()
        self.name = name
        self.reference = reference
        self.weight = weight

        # Initialize reaction
        self._rxn = AllChem.ReactionFromSmarts(self.smirks)
        self._rxn.Initialize()
        if self.num_reactant not in (1, 2):
            raise ValueError("Reaction is neither uni- nor bi-molecular.")

        # Extract reactants, agents, products
        reactants, agents, products = self.smirks.split(">")

        self.reactant_template = tuple(reactants.split("."))

        self.product_template = products
        self.agent_template = agents
        self.available_reactants = available_reactants

    def __repr__(self) -> str:
        return f"Reaction(smarts='{self.smirks}')"

    @property
    def num_reactant(self) -> Literal[1, 2]:
        return self.rxn.GetNumReactantTemplates()

    @property
    def num_agent(self) -> int:
        return self.rxn.GetNumAgentTemplates()

    @property
    def num_product(self) -> int:
        return self.rxn.GetNumProductTemplates()

    @property
    def rxn(self) -> Chem.rdChemReactions.ChemicalReaction:
        return self._rxn

    @classmethod
    def from_dict(cls, attrs: dict[str, Any]) -> Self:
        """Populate all attributes of the `Reaction` object from a dictionary."""
        if "smirks" in attrs:
            attrs["template"] = attrs.pop("smirks")
        return cls(**attrs)

    def to_dict(self) -> dict[str, Any]:
        """Returns serializable fields as new dictionary mapping."""

        out = {
            "template": self.smirks,
            "name": self.name,
            "reference": self.reference,
            "weight": self.weight,
            "available_reactants": self.available_reactants,
        }
        return out

    @functools.lru_cache(maxsize=20_000)
    def get_mol(self, smi: Union[str, Chem.rdchem.Mol]) -> Chem.rdchem.Mol:
        """Convert smiles to  `RDKit.Chem.rdchem.Mol`."""
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.rdchem.Mol):
            return smi
        else:
            raise TypeError(
                f"{type(smi)} not supported, only `str` or `Chem.rdchem.Mol`"
            )

    def get_smiles(self, mol: Union[str, Chem.rdchem.Mol]) -> str:
        """Convert `Chem.rdchem.Mol` to SMILES `str`."""
        if isinstance(mol, str):
            return mol
        elif isinstance(mol, Chem.rdchem.Mol):
            return Chem.MolToSmiles(mol)
        else:
            raise TypeError(
                f"{type(mol)} not supported, only `str` or `Chem.rdchem.Mol`"
            )

    def to_image(self, size: tuple[int, int] = (800, 300)) -> bytes:
        """Returns a png image of the visual represenation for this chemical reaction.

        Usage:
            * In Jupyter:

                >>> from IPython.display import Image
                >>> img = rxn.to_image()
                >>> Image(img)

            * save as image:

                >>> img = rxn.to_image()
                >>> pathlib.Path("out.png").write_bytes(img)

        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(*size)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        image = d2d.GetDrawingText()
        return image

    def is_reactant(self, smi: Union[str, Chem.rdchem.Mol]) -> bool:
        """Checks if `smi` is a reactant of this reaction."""
        mol = self.get_mol(smi)
        return self.rxn.IsMoleculeReactant(mol)

    def is_agent(self, smi: Union[str, Chem.rdchem.Mol]) -> bool:
        """Checks if `smi` is an agent of this reaction."""
        mol = self.get_mol(smi)
        return self.rxn.IsMoleculeAgent(mol)

    def is_product(self, smi: str) -> bool:
        """Checks if `smi` is a product of this reaction."""
        mol = self.get_mol(smi)
        return self.rxn.IsMoleculeProduct(mol)

    def is_reactant_first(self, smi: Union[str, Chem.rdchem.Mol]) -> bool:
        """Check if `smi` is the first reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, smi: Union[str, Chem.rdchem.Mol]) -> bool:
        """Check if `smi` the second reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def _check_smarts_match(
        self,
        first_mol: Chem.Mol,
        second_mol: Chem.Mol,
    ) -> bool:
        """Check if the reactants match the reaction template."""
        reactand_smarts_1 = Chem.MolFromSmarts(self.reactant_template[0])
        reactand_smarts_2 = Chem.MolFromSmarts(self.reactant_template[1])
        if not first_mol.HasSubstructMatch(reactand_smarts_1):
            return False
        if not second_mol.HasSubstructMatch(reactand_smarts_2):
            return False
        return True

    def can_run_reaction(
        self,
        first_reactant: str,
        second_reactant: Optional[str],
    ) -> bool:
        """Check if this reaction can be run with the given reactants.

        Parameters
        ----------
        first_reactant : str
            The first reactant for this reaction.
        second_reactant : Optional[str]
            The second reactant for this reaction.

        Returns
        -------
        bool
            True if the reaction can be run with the given reactants, False otherwise.
        """
        # Check first order reactions
        if self.num_reactant == 1:
            if second_reactant is not None:
                return False
            return self.is_reactant(first_reactant)

        # Check second order reactions
        if self.num_reactant != 2:
            raise ValueError(
                "Reaction is neither uni- or bi-molecular, cannot check for reactants."
            )
        if second_reactant is None:
            return False

        first_reactant_mol = self.get_mol(first_reactant)
        second_reactant_mol = self.get_mol(second_reactant)
        works_forward = self._check_smarts_match(
            first_reactant_mol, second_reactant_mol
        )
        works_backward = self._check_smarts_match(
            second_reactant_mol, first_reactant_mol
        )
        return works_forward or works_backward

    def run_reaction(
        self,
        first_reactant: str,
        second_reactant: Optional[str],
    ) -> Union[str, None]:
        """Run this reactions with reactants and return corresponding product.

        Par
            reactants (tuple): Contains SMILES strings for the reactants.

        Returns:
            uniqps: SMILES string representing the product or `None` if not reaction possible
        """
        if not self.can_run_reaction(first_reactant, second_reactant):
            raise ValueError("Reaction cannot be run with given reactants.")

        # Convert all reactants to `Chem.rdchem.Mol`
        reactant_list = [self.get_mol(first_reactant)]
        if second_reactant is not None:
            reactant_list.append(self.get_mol(second_reactant))
        r = reactant_list
        if len(r) > 1 and not self._check_smarts_match(*r):
            r = list(reversed(r))
        # Run reaction with rdkit
        reaction_result_list = self.rxn.RunReactants(tuple(r))
        single_product_reactions = []
        for reaction_result in reaction_result_list:
            product_list = {Chem.MolToSmiles(p) for p in reaction_result}
            if len(product_list) == 1:
                single_product_reactions.append(product_list.pop())

        if not single_product_reactions:
            raise FailedReconstructionError(f"No reaction with a single  {self.smirks}")
        product_smiles = sorted(single_product_reactions)[0]

        mol = Chem.MolFromSmiles(product_smiles)
        if Chem.MolFromSmiles(product_smiles) is None:
            raise FailedReconstructionError(
                f"Invalide product {product_smiles} for {self.smirks} with {first_reactant} and {second_reactant}"
            )
        return Chem.MolToSmiles(mol, isomericSmiles=False)

    def _filter_reactants(
        self, smiles: list[str], verbose: bool = False
    ) -> tuple[list[str], list[str]]:
        """Filters reactants which do not match the reaction."""
        smiles_iterator = tqdm(smiles) if verbose else smiles

        if self.num_reactant == 1:  # uni-molecular reaction
            reactants_1 = [
                smi for smi in smiles_iterator if self.is_reactant_first(smi)
            ]
            return reactants_1, []

        elif self.num_reactant == 2:  # bi-molecular reaction
            reactants_1 = [
                smi for smi in smiles_iterator if self.is_reactant_first(smi)
            ]
            reactants_2 = [
                smi for smi in smiles_iterator if self.is_reactant_second(smi)
            ]

            return reactants_1, reactants_2

        raise AssertionError("Reaction is neither uni- or bi-molecular!")

    def set_available_reactants(
        self, building_blocks: list[str], verbose: bool = False
    ) -> Self:
        """Finds applicable reactants from a list of building blocks.
        Sets `self.available_reactants`.
        """
        _available_reactants = self._filter_reactants(building_blocks, verbose=verbose)
        # Ensure molecules are stored as `str`
        _avail_r1 = [self.get_smiles(mol) for mol in _available_reactants[0]]
        if len(_avail_r1) == 0:
            logging.warning(f"No first reactants available for {self.smirks}")
        if self.num_reactant == 1:
            self.available_reactants = (_avail_r1,)
            return self
        if self.num_reactant == 2:
            _avail_r2 = [self.get_smiles(mol) for mol in _available_reactants[1]]
            if len(_avail_r2) == 0:
                logging.warning(f"No second reactants available for {self.smirks}")
            self.available_reactants = (_avail_r1, _avail_r2)
            return self

        raise NotImplementedError(
            f"Reaction {self.smirks} is neither uni- or bi-molecular!"
        )

    def has_available_reactants(self) -> bool:
        """Returns False if no reactants are available for any reactant position otherwise True."""
        if self.available_reactants is None:
            return False
        for reactants in self.available_reactants:
            if len(reactants) == 0:
                return False
        return True

    @property
    def get_available_reactants(self) -> set[str]:
        """Returns a set of all available reactants."""
        if self.available_reactants is None:
            return set()
        return {x for reactants in self.available_reactants for x in reactants}


class ReactionSet:
    """Represents a collection of reactions, for saving and loading purposes."""

    rxns: list[Reaction]

    def __init__(self, rxns: Optional[list[Reaction]] = None):
        self.rxns = rxns or []

    def __repr__(self) -> str:
        return f"ReactionSet ({len(self.rxns)} reactions.)"

    def __len__(self) -> int:
        return len(self.rxns)

    def __getitem__(self, index: int) -> Reaction:
        if self.rxns is None:
            raise IndexError("No Reactions.")
        return self.rxns[index]

    @classmethod
    def load(cls, file: PathType) -> Self:
        """Load a collection of reactions from a `*.json.gz` file."""

        if not str(file).endswith(".json.gz"):
            raise ValueError(f"Incompatible file extension for file {file}")

        with gzip.open(file, "r") as f:
            data = json.loads(f.read().decode("utf-8"))

        reactions = [Reaction.from_dict(_rxn) for _rxn in data["reactions"]]
        return cls(reactions)

    def save(self, file: str, skip_without_building_block: bool = True) -> None:
        """Save a collection of reactions to a `*.json.gz` file."""

        assert str(file).endswith(
            ".json.gz"
        ), f"Incompatible file extension for file {file}"
        if skip_without_building_block:
            reaction_list_to_save = [
                r for r in self.rxns if r.has_available_reactants()
            ]
        else:
            reaction_list_to_save = self.rxns
        rxns_as_json = {"reactions": [r.to_dict() for r in reaction_list_to_save]}
        with gzip.open(file, "w") as f:
            f.write(json.dumps(rxns_as_json, indent=4).encode("utf-8"))

    def _print(self, n: int = 3) -> None:
        """Debugging-helper method to print `n` reactions as json"""
        for i, r in enumerate(self.rxns):
            if i >= n:
                break
            print(json.dumps(r.to_dict(), indent=2))

    @property
    def num_unimolecular(self) -> int:
        return sum([r.num_reactant == 1 for r in self.rxns])

    @property
    def num_bimolecular(self) -> int:
        return sum([r.num_reactant == 2 for r in self.rxns])


# the definition of classes for defining synthetic trees below
@dataclass
class NodeChemical:
    """Represents a chemical node in a synthetic tree.

    Attributes:
        smiles (str): Molecule represented as SMILES string.
        parent (Optional[int]): Parent molecule represented as SMILES string (i.e. the result of a reaction)
        child (Optional[int]): Index of the reaction this object participates in.
        is_leaf (bool): Is this a leaf node in a synthetic tree?
        is_root (bool): Is this a root node in a synthetic tree?
        depth (float): Depth this node is in tree (+1 for an action, +.5 for a reaction)
        index (int): Incremental index for all chemical nodes in the tree.
    """

    smiles: str
    parent: Optional[int] = None
    child: Optional[int] = None
    is_leaf: bool = False
    is_root: bool = False
    depth: float = 0
    index: int = 0


@dataclass
class NodeRxn:
    """Represents a chemical reaction in a synthetic tree.


    Attributes
    ---------
        rxn_id:
            Index to a reaction lookup table.
        rtype:
            Indicator for uni (1) or bi-molecular (2) reaction.
        parent: str
            Product of this reaction.
        child: str
            Reactants for this reaction.
        depth:
            Depth this node is in tree (+1 for an action, +.5 for a reaction)
        index:
            Order of this `NodeRxn` in a `SyntheticTree`.
    """

    rxn_id: int
    rtype: Optional[int] = None
    parent: Optional[str] = None
    child: Optional[list[str]] = None
    depth: float = 0
    index: int = 0


class SyntheticTree:
    """Representation of a synthetic tree (syntree).

    Attributes:
    chemicals: list[NodeChemical]
        A list of chemical nodes, in order of addition.
    reactions: list[NodeRxn]
        A list of reaction nodes, in order of addition.
    root: NodeChemical
        The root node of the tree.
    depth: float
        The depth of the tree.
    actions: list[int]
        A list of actions, in order of addition.
    rxn_id2type: Optional[dict]
        A dictionary mapping reaction ids to reaction types.
    ACTIONS: dict[int, str]
        A dictionary mapping action ids to action types.
    """

    chemicals: list[NodeChemical]
    reactions: list[NodeRxn]
    root: Optional[NodeChemical]
    depth: float
    actions: list[int]
    action_mapping: dict[int, str]

    def __init__(self) -> None:
        """Initialize an empty SyntheticTree."""
        self.chemicals = []
        self.reactions = []
        self.root = None
        self.depth = 0
        self.actions = []
        self.action_mapping = {
            0: "add",
            1: "expand",
            2: "merge",
            3: "end",
        }

    def __repr__(self) -> str:
        return f"SynTree(num_actions={self.num_actions})"  # This is including the end action

    @classmethod
    def from_dict(cls, attrs: dict[str, Any]) -> Self:
        """Initialize a `SyntheticTree` from a dictionary."""
        syntree = cls()
        syntree.root = NodeChemical(**attrs["root"]) if attrs["root"] else None
        syntree.depth = attrs["depth"]
        syntree.actions = attrs["actions"]

        syntree.reactions = [NodeRxn(**_rxn_dict) for _rxn_dict in attrs["reactions"]]
        syntree.chemicals = [
            NodeChemical(**_chem_dict) for _chem_dict in attrs["chemicals"]
        ]
        return syntree

    def to_dict(self) -> dict[str, Any]:
        """Export this `SyntheticTree` to a dictionary."""
        return {
            "reactions": [r.__dict__ for r in self.reactions],
            "chemicals": [m.__dict__ for m in self.chemicals],
            "root": self.root.__dict__ if self.root else None,
            "depth": self.depth,
            "actions": self.actions,
        }

    def _print(self) -> None:
        """Print the contents of this `SyntheticTree`."""
        print(f"============SynTree (depth={self.depth:>4.1f})==============")
        print("===============Stored Molecules===============")
        for compound_node in self.chemicals:
            suffix = " (root mol)" if compound_node.is_root else ""
            print(compound_node.smiles, suffix)
        print("===============Stored Reactions===============")
        for reaction_node in self.reactions:
            print(
                f"{reaction_node.rxn_id} ({'bi ' if reaction_node.rtype==2 else 'uni'})"
            )
        print("===============Followed Actions===============")
        print(self.actions)
        print("==============================================")

    def get_node_index(self, smi: str) -> Optional[int]:
        """Return the index of the node matching the input SMILES.

        If the query molecule is not in the tree, return None.
        """
        # Info: reversed() is a prelim fix for a bug that caused three mols in the state!
        for node in reversed(self.chemicals):
            if smi == node.smiles:
                return node.index
        return None

    def get_state(self) -> tuple[Optional[str], Optional[str]]:
        """Get the state of this synthetic tree.
        The most recent root node has 0 as its index.

        Returns
        -------
        state: list[str]
            A list contains all root node molecules.
        """
        root_state_list = [node.smiles for node in self.chemicals if node.is_root][::-1]
        if len(root_state_list) == 0:
            return None, None
        if len(root_state_list) == 1:
            return root_state_list[0], None
        if len(root_state_list) == 2:
            return root_state_list[0], root_state_list[1]
        raise AssertionError("There should be at most two root nodes.")

    def merge(
        self, mol1: str, mol2: Optional[str], rxn_id: int, mol_product: str
    ) -> None:
        if mol1 is None or mol2 is None:
            raise AssertionError("Merging requires two molecules.")
        mol1_idx = self.get_node_index(mol1)
        mol2_idx = self.get_node_index(mol2)
        if mol1_idx is None:
            raise AssertionError(f"Cannot find {mol1} in the tree.")
        if mol2_idx is None:
            raise AssertionError(f"Cannot find {mol2} in the tree.")

        node_mol1 = self.chemicals[mol1_idx]
        node_mol2 = self.chemicals[mol2_idx]

        node_rxn = NodeRxn(
            rxn_id=rxn_id,
            rtype=2,
            parent=None,
            child=[node_mol1.smiles, node_mol2.smiles],
            depth=max(node_mol1.depth, node_mol2.depth) + 0.5,
            index=len(self.reactions),
        )
        node_product = NodeChemical(
            smiles=mol_product,
            parent=None,
            child=node_rxn.rxn_id,
            is_leaf=False,
            is_root=True,
            depth=node_rxn.depth + 0.5,
            index=len(self.chemicals),
        )

        node_rxn.parent = node_product.smiles
        node_mol1.parent = node_rxn.rxn_id
        node_mol2.parent = node_rxn.rxn_id
        node_mol1.is_root = False
        node_mol2.is_root = False

        self.chemicals.append(node_product)
        self.reactions.append(node_rxn)

    def end(self) -> None:
        """End this synthetic tree."""
        self.root = self.chemicals[-1]
        self.depth = self.root.depth

    def extend(
        self, mol1: Optional[str], mol2: Optional[str], rxn_id: int, mol_product: str
    ) -> None:
        if mol1 is None:
            raise AssertionError("Molecule cannot be None.")
        mol1_idx = self.get_node_index(mol1)
        if mol1_idx is None:
            raise AssertionError(f"Cannot find {mol1} in the tree.")

        if mol2 is None:
            node_mol1 = self.chemicals[mol1_idx]
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=1,
                parent=None,
                child=[node_mol1.smiles],
                depth=node_mol1.depth + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals),
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif mol2 is not None:  # Expand with bi-mol rxn
            if mol1 is None:
                raise AssertionError("Molecule cannot be None.")
            mol1_idx = self.get_node_index(mol1)
            if mol1_idx is None:
                raise AssertionError(f"Cannot find {mol1} in the tree.")
            node_mol1 = self.chemicals[mol1_idx]
            node_mol2 = NodeChemical(
                smiles=mol2,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=max(node_mol1.depth, node_mol2.depth) + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals) + 1,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

    def update(
        self,
        action: int,
        rxn_id: Optional[int],
        mol1: Optional[str],
        mol2: Optional[str],
        mol_product: Optional[str],
    ) -> None:
        """Update this synthetic tree by adding a reaction step.

        Info:
            Recall that actions "add", "expand", "merge" have a reaction, "end" does not.
            Hence, the following 6 updates are possible:
                - Action: End
                - Action: Add
                    - with unimolecular reaction
                    - with bimolecular reaction
                - Action: Expand
                    - with unimolecular reaction
                    - with bimolecular reaction
                - Action: Merge
                    - with bimolecular reaction

        Parameters
        ----------
        action: int
            action_id corresponding to the action taken. (ref `self.ACTIONS`)
        rxn_id: int
            id of the reaction
        mol1: str
            First reactant as SMILES-string
        mol2: str
            Second reactant as SMILES-string
        mol_product: str
            Product of the reaction as SMILES-string

        Returns
        -------
        None
        """
        self.actions.append(int(action))

        if action == 3:  # End
            self.end()
            self.depth = max([node.depth for node in self.reactions]) + 0.5
            return None

        if mol1 is None:
            raise AssertionError("mol1 cannot be None.")
        if mol_product is None:
            raise AssertionError("mol_product cannot be None.")
        if rxn_id is None:
            raise AssertionError("rxn_id cannot be None.")

        if action == 2:  # Merge (with bi-mol rxn)
            self.merge(mol1, mol2, rxn_id, mol_product)

        elif action == 1:  # Expand with uni-mol rxn
            self.extend(mol1, mol2, rxn_id, mol_product)

        elif action == 0 and mol2 is None:  # Add with uni-mol rxn
            node_mol1 = NodeChemical(
                smiles=mol1,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=1,
                parent=None,
                child=[node_mol1.smiles],
                depth=0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=1,
                index=len(self.chemicals) + 1,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is not None:  # Add with bi-mol rxn
            node_mol1 = NodeChemical(
                smiles=mol1,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_mol2 = NodeChemical(
                smiles=mol2,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals) + 1,
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=1,
                index=len(self.chemicals) + 2,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        else:
            raise ValueError("Check input")
        self.depth = max([node.depth for node in self.reactions]) + 0.5
        return None

    @property
    def nodes_as_smiles(self) -> list[str]:
        """Returns all (leaf, inner, root) molecules in this tree as smiles"""
        return [node.smiles for node in self.chemicals]

    @property
    def leafs_as_smiles(self) -> list[str]:
        """Returns all leaf molecules in this tree as smiles"""
        return [node.smiles for node in self.chemicals if node.is_leaf]

    @property
    def nonleafs_as_smiles(self) -> list[str]:
        """Returns all non-leaf (inner + root) molecules in this tree as smiles"""
        return [node.smiles for node in self.chemicals if not node.is_leaf]

    @property
    def is_valid(self) -> bool:
        """Valid if it has "actions" and has been ended properly with "end"-action"""
        return self.num_actions > 0 and self.actions[-1] == 3

    @property
    def num_actions(self) -> int:
        """Number of actions
        Info:
            The depth of a tree is not a perfect metric for complexity,
            as a 2nd subtree that gets merged only increases depth by 1.
        """
        return len(self.actions)


class SyntheticTreeSet:
    """Represents a collection of synthetic trees, for saving and loading purposes."""

    synthetic_tree_list: list[SyntheticTree]

    def __init__(self, sts: Optional[list[SyntheticTree]] = None):
        if sts is not None:
            self.synthetic_tree_list = sts
        else:
            self.synthetic_tree_list = []

    def __repr__(self) -> str:
        return f"SyntheticTreeSet ({len(self.synthetic_tree_list)} syntrees.)"

    def __len__(self) -> int:
        return len(self.synthetic_tree_list)

    def __getitem__(self, index: int) -> SyntheticTree:
        if self.synthetic_tree_list is None:
            raise IndexError("No Synthetic Trees.")
        return self.synthetic_tree_list[index]

    @classmethod
    def load(cls, file: PathType) -> SyntheticTreeSet:
        """Load a collection of synthetic trees from a `*.json.gz` file."""
        assert str(file).endswith(
            ".json.gz"
        ), f"Incompatible file extension for file {file}"

        with gzip.open(file, "rt") as f:
            data = json.loads(f.read())

        syntrees = [SyntheticTree.from_dict(_syntree) for _syntree in data["trees"]]

        return cls(syntrees)

    def save(self, file: str) -> None:
        """Save a collection of synthetic trees to a `*.json.gz` file."""
        assert str(file).endswith(
            ".json.gz"
        ), f"Incompatible file extension for file {file}"

        syntrees_as_json = {
            "trees": [st.to_dict() for st in self.synthetic_tree_list if st is not None]
        }
        with gzip.open(file, "wt") as f:
            f.write(json.dumps(syntrees_as_json))

    def split_by_depth(self) -> dict[int, list[SyntheticTree]]:
        """Splits syntrees by depths and returns a copy."""
        trees_by_depth_dict: dict[int, list[SyntheticTree]] = {}
        for st in self.synthetic_tree_list:
            depth = int(st.depth)
            if depth not in trees_by_depth_dict:
                trees_by_depth_dict[depth] = []
            trees_by_depth_dict[depth].append(st)
        return trees_by_depth_dict

    def _print(self, x: int = 3) -> None:
        """Helper function for debugging."""
        for i, r in enumerate(self.synthetic_tree_list):
            if i >= x:
                break
            print(r.to_dict())
