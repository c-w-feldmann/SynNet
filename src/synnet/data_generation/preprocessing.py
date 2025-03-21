"""Functions for preprocessing building blocks and reaction templates."""

from functools import partial
from pathlib import Path
from typing import Iterable, List, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools, rdMolDescriptors

try:
    from pathos import multiprocessing as mp
except ImportError:
    logger.warning("Failed to import pathos, using multiprocessing instead.")
    import multiprocessing as mp

from synnet.config import MAX_PROCESSES
from synnet.utils.custom_types import PathType
from synnet.utils.data_utils import Reaction
from synnet.utils.parallel import chunked_parallel


def parse_sdf_file(file: str) -> pd.DataFrame:
    """Parse `*.sdf` file and return a pandas dataframe."""
    df = PandasTools.LoadSDF(
        file,
        idName="ID",
        molColName=None,
        includeFingerprints=False,
        isomericSmiles=False,
        smilesName="raw_smiles",
        embedProps=False,
        removeHs=True,
        strictParsing=True,
    )

    # Supplier specifies available building blocks with an "X".
    # Let us convert this to a boolean.
    availability_cols = [col for col in df.columns if col.startswith("avail")]
    df[availability_cols] = df[availability_cols].apply(lambda x: x == "X")

    # Canonicalize Smiles
    df["SMILES"] = df["raw_smiles"].apply(
        lambda x: Chem.MolToSmiles(
            Chem.MolFromSmiles(x), canonical=True, isomericSmiles=False
        )
    )
    return df


class BuildingBlockFilter:
    """Filter building blocks."""

    building_blocks_filtered: list[str] = []
    rxns_initialised: bool = False

    def __init__(
        self,
        *,
        building_blocks: list[Union[str, AllChem.rdchem.Mol]],
        rxn_templates: list[str],
        processes: int = MAX_PROCESSES,
        verbose: bool = False,
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates

        # Init reactions
        self.rxns = [Reaction(template=template) for template in self.rxn_templates]

        self.processes = processes
        self.verbose = verbose

    def _match_mp(self) -> Self:
        def __match(_rxn: Reaction, *, bblocks: list[str]) -> Reaction:
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, bblocks=self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            self.rxns = pool.map(func, self.rxns)
        return self

    def _filter_bblocks_for_rxns(self) -> Self:
        """Initializes a `Reaction` with a list of possible reactants."""

        if self.processes == 1:
            self.rxns = [
                rxn.set_available_reactants(self.building_blocks) for rxn in self.rxns
            ]
        else:
            self._match_mp()

        self.rxns_initialised = True
        return self

    def filter(self) -> Self:
        """Filters out building blocks which do not match a reaction template."""
        if not self.rxns_initialised:
            self._filter_bblocks_for_rxns()

        matched_bblocks = {x for rxn in self.rxns for x in rxn.get_available_reactants}
        self.building_blocks_filtered = list(matched_bblocks)
        return self


class BuildingBlockFileHandler:
    """Handler for building blocks files."""

    def load(self, file: PathType) -> list[str]:
        """Load building blocks from file."""
        return pd.read_csv(file)["SMILES"].to_list()

    def save(self, file: PathType, building_blocks: list[str]) -> None:
        """Save building blocks to file."""
        if not isinstance(file, Path):
            file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"SMILES": building_blocks})
        df.to_csv(file, index=False)


class ReactionTemplateFileHandler:
    def load(self, file: str) -> list[str]:
        """Load reaction templates from file."""
        with open(file, "rt") as f:
            rxn_templates = f.readlines()

        rxn_templates = [tmplt.strip() for tmplt in rxn_templates]

        if not all([self._validate(t)] for t in rxn_templates):
            raise ValueError("Not all reaction templates are valid.")

        return rxn_templates

    def save(self, file: str, rxn_templates: list[str]) -> None:
        """Save reaction templates to file."""
        with open(file, "wt") as f:
            f.writelines(t + "\n" for t in rxn_templates)

    def _validate(self, rxn_template: str) -> bool:
        """Validate reaction templates.

        Checks if:
          - reaction is uni- or bimolecular
          - has only a single product

        Note:
          - only uses std-lib functions, very basic validation only
        """
        reactants, agents, products = rxn_template.split(">")
        is_uni_or_bimolecular = len(reactants) == 1 or len(reactants) == 2
        has_single_product = len(products) == 1

        return is_uni_or_bimolecular and has_single_product


class BuildingBlockFilterHeuristics:
    @staticmethod
    def filter(
        bblocks: Iterable[str], return_as: str = "list", verbose: bool = False
    ) -> Union[pd.DataFrame, List[str]]:
        """Filter building blocks based on heuristics.

        See: https://doi.org/10.1021/acs.jcim.2c00785, SI Figure 12)
        """
        # Convert bblocks to DataFrame for convenience
        df = pd.DataFrame(bblocks, columns=["SMILES"])
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol="SMILES", molCol="mol")

        # Compute properties
        functions = [
            rdMolDescriptors.CalcNumHeavyAtoms,
            rdMolDescriptors.CalcNumAmideBonds,
            rdMolDescriptors.CalcNumRings,
            rdMolDescriptors.CalcFractionCSP3,
            rdMolDescriptors.CalcExactMolWt,
            rdMolDescriptors.CalcNumRotatableBonds,
        ]

        for func in functions:
            name = func.__name__.removeprefix("Calc")
            df[name] = df["mol"].apply(func)

        # Filter based on heuristics
        idx_remove = (
            (df["SMILES"].isna())  # (df["NumHeavyAtoms"] < 5) \
            | (df["NumHeavyAtoms"] > 40)
            | (df["NumRotatableBonds"] > 16)
            | (df["NumAmideBonds"] > 5)
            | (df["NumHeavyAtoms"] > 40)
        )

        if verbose:
            n_total = len(df)
            n_keep = len(df) - idx_remove.sum()
            logger.info("Filtering building blocks based on heuristics:")
            logger.info(f"  Total number of building blocks {n_total:d}")
            logger.info(
                f"  Retained number of building blocks {n_keep:d} ({n_keep/n_total:.2%})"
            )

        if return_as == "list":
            return df.loc[~idx_remove, "SMILES"].tolist()
        elif return_as == "df":
            df["idx_remove"] = idx_remove
            return df
        else:
            return df.loc[~idx_remove]


class BuildingBlockFilterMatchRxn:
    def filter(
        self,
        bblocks: Iterable[str],
        rxn_templates: Iterable[str],
        *,
        ncpu: int = MAX_PROCESSES,
        verbose: bool = False,
    ) -> tuple[List[str], List[Reaction]]:
        """Filter building blocks based on a match to a reaction template.
        If a building block matches a reaction template, it is retained.

        Return:
            matched_bblocks: List[str] - list of building blocks that matched a reaction template
            reactions: List[Reaction] - initialized reactions
        """
        # Match building blocks to reactions
        logger.info("Converting SMILES to `rdkit.Mol` objects...")
        bblocks = chunked_parallel(
            list(bblocks), lambda x: AllChem.MolFromSmiles(x), verbose=verbose
        )

        logger.info("Converting reaction templates to `rdkit.Reaction` objects...")
        reactions = [Reaction(tmpl) for tmpl in rxn_templates]

        logger.info("Matching building blocks to reactions...")
        func = partial(self.match_bblocks, building_blocks=bblocks)
        reaction_list: list[Reaction] = chunked_parallel(
            reactions, func, verbose=verbose, max_cpu=ncpu, chunks=9
        )

        matched_bblocks: List[str] = list(
            {x for rxn in reaction_list for x in rxn.get_available_reactants}
        )

        if verbose:
            n_total = len(list(bblocks))
            n_keep = len(matched_bblocks)
            logger.info(
                "Filtering building blocks based on match to reaction templates:"
            )
            logger.info(f"  Total number of building blocks {n_total:d}")
            logger.info(
                f"  Retained number of building blocks {n_keep:d} ({n_keep / n_total:.2%})"
            )

        return matched_bblocks, reaction_list

    @staticmethod
    def match_bblocks(
        reaction: Reaction,
        *,
        building_blocks: list[Union[str, AllChem.rdchem.Mol]],
    ) -> Reaction:
        return reaction.set_available_reactants(building_blocks)
