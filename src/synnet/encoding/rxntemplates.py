"""Reaction fingerprints with RDKit."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable, Union

import numpy as np
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdChemReactions import (
    CreateDifferenceFingerprintForReaction,
    CreateStructuralFingerprintForReaction,
    FingerprintType,
    ReactionFingerprintParams,
)

from synnet.data_generation.syntrees import Encoder

ChemicalReaction = rdkit.Chem.rdChemReactions.ChemicalReaction


@dataclass(frozen=True)
class RdkitRxnFPConfig:
    """Parameters for `rdkit.Chem.rdChemReactions.ReactionFingerprintParams`.

    NOTE: Do not change order! Only positional args allowed.

    Ref: https://rdkit.org/docs/cppapi/structRDKit_1_1ReactionFingerprintParams.html
    """

    includeAgents: bool = False
    bitRatioAgents: float = 0.2
    nonAgentWeight: int = 10
    agentWeight: int = 1
    fpSize: int = 2048
    fpType: str = "AtomPairFP"

    # Post init, check that fptype is valid
    def __post_init__(self):
        valid_fptypes = ["AtomPairFP", "TopologicalTorsion", "MorganFP", "RDKitFP", "PatternFP"]
        if self.fpType not in valid_fptypes:
            raise ValueError(f"Invalid fptype: {self.fpType}. Must be one of {valid_fptypes}")

    @property
    def params(self) -> dict:
        return asdict(self)

    def get_config(self) -> ReactionFingerprintParams:
        fpType: Callable = FingerprintType.names[self.fpType]
        kwargs = asdict(self)
        kwargs.update({"fpType": fpType})
        args = list(kwargs.values())
        return ReactionFingerprintParams(*args)  # only positional args allowed


class RXNFingerprintEncoder(Encoder):
    def __init__(
        self,
        mode: str = "structural",
        params: dict = RdkitRxnFPConfig().params,
        rxn_map: dict[int, str] = None,  # to map int to reaction smiles
    ) -> None:
        super().__init__()
        # Input validation
        valid_modes = ["difference", "structural"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # Get rdkit fct corresponding to mode
        self.mode = mode
        if mode == "structural":
            self._fct = CreateStructuralFingerprintForReaction
        elif mode == "difference":
            # NOTE: Difference fps may not be in [0, 1], so we default to structural fps
            self._fct = CreateDifferenceFingerprintForReaction
        # Store params and get ReactionFingerprintParams
        self.params = params
        self._rdkitparams = RdkitRxnFPConfig(**params).get_config()

        # Reaction map (helper for creating dataset)
        self.rxn_map = rxn_map

    @lru_cache(maxsize=128)
    def _from_string(self, input: str) -> ChemicalReaction:
        """Convert a reaction smiles/smirks/smarts to a rdkit ChemicalReaction object"""
        return Chem.ReactionFromSmarts(input)

    def _encode_to_bv(
        self, rxn: ChemicalReaction
    ) -> rdkit.DataStructs.cDataStructs.UIntSparseIntVect:
        return self._fct(rxn, self._rdkitparams)

    def encode(self, rxn: Union[ChemicalReaction, str]) -> np.ndarray:
        """Encode a reaction"""
        if isinstance(rxn, str):
            rxn = self._from_string(rxn)
        if isinstance(rxn, int):
            # We are creating a dataset.
            # For legacy reasons, we only have access to the reaction id in the syntree,
            # not the reaction smiles.
            rxn = self._from_string(self.rxn_map[rxn])

        # Compute fingerprint
        bv = self._encode_to_bv(rxn)

        # Convert to numpy array
        fp = np.zeros(0, dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(bv, fp)

        return fp

    def encode_batch(self, inputs: list[ChemicalReaction, str]) -> np.ndarray:
        """Encode a batch of reactions"""
        fps = [self.encode(x) for x in inputs]
        return np.asarray(fps)
