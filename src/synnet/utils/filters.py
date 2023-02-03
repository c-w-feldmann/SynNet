import logging
from typing import Optional, Union

import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tdc import Oracle

from synnet.config import MAX_PROCESSES
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel

logger = logging.getLogger()


# Note: Boolean indexing is faster than df.query() but this is more convenient.
FILTERS = {
    "all": "`num_actions`>0",  # dummy filter that always returns True
    "qed": "`qed` > @THRESHOLD",
    "small molecules": "(`NumHeavyAtoms` <= 40) & (`NumRotatableBonds` <= 16) & (`NumAmideBonds` <= 5)",
    "qed + random": "`random` < (`qed` / @THRESHOLD)",
    "qed + small molecules": "(`qed` > @THRESHOLD) & (`NumHeavyAtoms` <= 40) & (`NumRotatableBonds` <= 16) & (`NumAmideBonds` <= 5)",
    "qed + random + small molecules": "(`random` < (`qed` / @THRESHOLD)) & (`NumHeavyAtoms` <= 40) & (`NumRotatableBonds` <= 16) & (`NumAmideBonds` <= 5)",
}

FILTER_LABELS = list(FILTERS.keys())
FILTER_QUERIES = list(FILTERS.values())

assert len(FILTER_LABELS) == len(FILTER_QUERIES), "Each filter must have a label and a query"

# Per default, the first filter - "all" - is not really a filter.
# Hence, attribute a grey color to it.
COLORS = ["#cdd4d4"] + sns.color_palette("husl", len(FILTER_QUERIES) - 1).as_hex()


def demo_colors(COLORS: list[str]):
    """Demo the colors used for the filters."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 1))
    ax.bar(range(len(COLORS)), [1] * len(COLORS), color=COLORS)


RDKIT_MOL_DESCRIPTORS = [
    rdMolDescriptors.CalcNumHeavyAtoms,
    rdMolDescriptors.CalcNumAmideBonds,
    rdMolDescriptors.CalcNumRings,
    rdMolDescriptors.CalcFractionCSP3,
    rdMolDescriptors.CalcExactMolWt,
    rdMolDescriptors.CalcNumRotatableBonds,
]
TDC_MOL_ORACLES = [
    Oracle("qed"),
    Oracle("SA"),
    Oracle("LogP"),
]

# MOL_DESCRIPTORS: dict[str,Callable]= {
#     **{func.__name__.removeprefix("Calc"): func for func in RDKIT_MOL_DESCRIPTORS},
#     **{oracle.name: oracle for oracle in TDC_MOL_ORACLES},
# } # The only thing is that rdkit descriptors input is a mol and tdc oracles input is a smiles...


def calc_metrics_on_syntree_collection(
    syntree_coll: SyntheticTreeSet,
    smiles_column: str = "root_smiles",
    mol_column: Optional[str] = None,
    verbose: bool = True,
    max_cpu: int = MAX_PROCESSES,
) -> pd.DataFrame:
    """Calculate metrics on a collection of SyntheticTree objects.

    Args:
        syntree_coll (SyntheticTreeSet): Collection of SyntheticTree objects

    Returns:
        df (pd.DataFrame): DataFrame with metrics

    """
    # Compute metrics
    _root_mols_data = chunked_parallel(
        syntree_coll.sts, calc_metrics_on_mol, verbose=verbose, max_cpu=max_cpu
    )

    # Store all in df
    df = pd.DataFrame(syntree_coll.sts, columns=["syntrees"])
    final_df = pd.DataFrame.from_dict(_root_mols_data).rename(columns={"smi": smiles_column})
    final_df = pd.concat([df, final_df], axis=1)

    # Drop or rename mol column?
    if mol_column is not None:
        final_df.drop(columns=["mol"], inplace=True)
    else:
        final_df.rename(columns={"mol": mol_column}, inplace=True)

    return final_df


def calc_metrics_on_mol(x: Union[str, Chem.rdchem.Mol, SyntheticTree]) -> dict:
    """Calculate metrics on a single molecule.

    Args:
        x: a molecule (as SMILES, rdkit.Chem.rdChem.Mol or taken as root from SyntheticTree)

    Returns:
        mol_dict (dict): Dictionary with metrics

    """
    if isinstance(x, SyntheticTree):
        mol_dict = {
            "smi": x.root.smiles,
            "mol": Chem.MolFromSmiles(x.root.smiles),
        }
    elif isinstance(x, str):
        mol_dict = {
            "smi": x,
            "mol": Chem.MolFromSmiles(x),
        }
    elif isinstance(x, Chem.rdchem.Mol):
        mol_dict = {
            "smi": Chem.MolToSmiles(x),
            "mol": x,
        }

    # Compute mol descriptors (on rdkit.Chem.rdChem.Mol)
    mol_dict.update(
        {
            func.__name__.removeprefix("Calc"): func(mol_dict["mol"])
            for func in RDKIT_MOL_DESCRIPTORS
        }
    )
    # Compute QED (on str)
    mol_dict.update({oracle.name: oracle(mol_dict["smi"]) for oracle in TDC_MOL_ORACLES})

    return mol_dict
