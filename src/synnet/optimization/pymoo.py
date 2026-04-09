"""Module with pymoo solver class for optimization."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.population import Individual
from pymoo.core.problem import Problem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

DecoderFunction = Callable[[npt.NDArray[Any]], Optional[str]]
ScorerFunction = Callable[[Optional[str]], float]


class SmilesDuplicateElimination(ElementwiseDuplicateElimination):
    """Duplicate elimination based on SMILES."""

    decoder: DecoderFunction

    def __init__(self, decoder: DecoderFunction) -> None:
        """Initialize duplicate elimination by exact SMILES match.

        Parameters
        ----------
        decoder : DecoderFunction
            Function mapping latent vectors to SMILES strings.

        """
        self.decoder = decoder
        super().__init__()

    def _get_smiles(self, individual: Individual) -> str | None:
        """Get or lazily decode SMILES for an individual.

        Parameters
        ----------
        individual : Individual
            Population individual.

        Returns
        -------
        str | None
            Cached or decoded SMILES string.

        """
        if individual.has("smiles"):
            return individual.get("smiles")
        smiles = self.decoder(individual.X)
        individual.set("smiles", smiles)
        return smiles

    def is_equal(self, a: Individual, b: Individual) -> bool:
        """Check whether two individuals decode to identical SMILES.

        Parameters
        ----------
        a : Individual
            First individual.
        b : Individual
            Second individual.

        Returns
        -------
        bool
            ``True`` when both decoded SMILES are identical.

        """
        smiles_a = self._get_smiles(a)
        smiles_b = self._get_smiles(b)
        return smiles_a == smiles_b


class SimilarityDuplicateElimination(ElementwiseDuplicateElimination):
    """Duplicate elimination based on Tanimoto Similarity."""

    def __init__(self, decoder: DecoderFunction, threshold: float) -> None:
        """Initialize duplicate elimination by Tanimoto threshold.

        Parameters
        ----------
        decoder : DecoderFunction
            Function mapping latent vectors to SMILES strings.
        threshold : float
            Similarity threshold used to mark individuals as duplicates.

        """
        super().__init__()
        self.decoder = decoder
        self.threshold = threshold

    def _get_smiles(self, individual: Individual) -> str | None:
        """Get or lazily decode SMILES for an individual.

        Parameters
        ----------
        individual : Individual
            Population individual.

        Returns
        -------
        str | None
            Cached or decoded SMILES string.

        """
        if individual.has("smiles"):
            return individual.get("smiles")
        smiles = self.decoder(individual.X)
        individual.set("smiles", smiles)
        return smiles

    def is_equal(self, a: Individual, b: Individual) -> bool:
        """Check whether two individuals are duplicates by similarity.

        Parameters
        ----------
        a : Individual
            First individual.
        b : Individual
            Second individual.

        Returns
        -------
        bool
            ``True`` when Tanimoto similarity is at least ``threshold``.

        """
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
        limits: tuple[float | None, float | None] | None = None,
    ):
        """Initialize the optimization problem.

        Parameters
        ----------
        decoder : DecoderFunction
            Decoder mapping latent vectors to SMILES.
        oracle : ScorerFunction
            Objective scorer function.
        n_var : int
            Number of optimization variables.
        n_obj : int
            Number of objectives.
        limits : tuple[float | None, float | None] | None, optional
            Lower and upper bounds for optimization variables.

        """
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
        """Evaluate the problem.

        Parameters
        ----------
        embedded_representation: npt.NDArray[Any]
            The matrix of embedded vectors to evaluate.
        out: dict[str, Any]
            The dictionary to store the results in.
        *_args: Any
            Additional arguments.
        **_kwargs: Any
            Additional keyword arguments.
            Can contain "smiles_list", which is a list of SMILES strings corresponding
            to the embedded representation. If not provided, the SMILES will be
            generated using the decoder.

        """
        smiles_list: Optional[list[Optional[str]]]
        smiles_list = _kwargs.get("smiles_list", None)
        if smiles_list is None:
            smiles_list = [
                self.decoder(embedding) for embedding in embedded_representation
            ]
            out["smiles"] = smiles_list
        if len(smiles_list) != embedded_representation.shape[0]:
            raise AssertionError("Number of smiles and embeddings do not match.")
        out["F"] = np.array([self.oracle(smiles) for smiles in smiles_list])
