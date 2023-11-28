"""Module with pymoo solver class for optimization."""
from __future__ import annotations
from typing import Any, Callable, Optional

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.population import Individual
from pymoo.core.problem import Problem

import numpy as np
import numpy.typing as npt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import (
    TanimotoSimilarity
)


DecoderFunction = Callable[[npt.NDArray[Any]], Optional[str]]
ScorerFunction = Callable[[Optional[str]], float]


class SmilesDuplicateElimination(ElementwiseDuplicateElimination):
    """Duplicate elimination based on SMILES."""

    decoder: DecoderFunction

    def __init__(self, decoder: DecoderFunction) -> None:
        self.decoder = decoder
        super().__init__()

    def _get_smiles(self, individual: Individual) -> Optional[str]:
        if individual.has("smiles"):
            return individual.get("smiles")
        else:
            smiles = self.decoder(individual.X)
            individual.set("smiles", smiles)
            return smiles

    def is_equal(self, a: Individual, b: Individual) -> bool:
        smiles_a = self._get_smiles(a)
        smiles_b = self._get_smiles(b)
        return smiles_a == smiles_b


class SimilarityDuplicateElimination(ElementwiseDuplicateElimination):
    """Duplicate elimination based on Tanimoto Similarity."""

    def __init__(
        self,
        decoder: DecoderFunction,
        threshold: float
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.threshold = threshold

    def _get_smiles(self, individual: Individual) -> Optional[str]:
        if individual.has("smiles"):
            return individual.get("smiles")
        else:
            smiles = self.decoder(individual.X)
            individual.set("smiles", smiles)
            return smiles

    def is_equal(self, a: Individual, b: Individual) -> bool:
        mol_a = Chem.MolFromSmiles(self._get_smiles(a))
        mol_b = Chem.MolFromSmiles(self._get_smiles(b))
        if mol_a is None or mol_b is None:
            return True
        bit_vector_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2)
        bit_vector_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2)
        tanimoto = TanimotoSimilarity(bit_vector_a, bit_vector_b)
        return tanimoto >= self.threshold


class SmilesGenerationProblem(Problem):
    """The default pymoo chemspace problem."""

    decoder: DecoderFunction
    oracle: ScorerFunction

    def __init__(
        self,
        decoder: DecoderFunction,
        oracle: ScorerFunction,
        n_var: int,
        n_obj: int,
        limits: Optional[tuple[Optional[float], Optional[float]]] = None,
    ):
        if limits is None:
            limits = (None, None)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=limits[0],
            xu=limits[1],
            requires_kwargs=True,  # Otherwise, the smiles are not passed to the evaluate function
        )
        self.decoder = decoder
        self.oracle = oracle

    def _evaluate(
        self,
        embedded_representation: npt.NDArray[Any],
        out: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        smiles_list: Optional[list[Optional[str]]]
        smiles_list = _kwargs.get("smiles_list", None)
        if smiles_list is None:
            smiles_list = [self.decoder(embedding) for embedding in embedded_representation]
            out["smiles"] = smiles_list
        if len(smiles_list) != embedded_representation.shape[0]:
            raise AssertionError("Number of smiles and embeddings do not match.")
        out["F"] = np.array([self.oracle(smiles) for smiles in smiles_list])
