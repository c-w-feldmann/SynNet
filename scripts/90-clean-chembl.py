"""Clean chembl data.

Info:
    Some SMILES in the chembl data are invalid.
    We clean all SMILES by converting them from `smiles`-> `mol`->`smiles`.
"""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

import pandas as pd
from loguru import logger
from rdkit import Chem


def smiles2smiles(smiles: str) -> str | None:
    """Convert smiles -> mol -> smiles.

    Parameters
    ----------
    smiles : str
        A SMILES string.

    Returns
    -------
    str or None
        A valid SMILES string or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the SMILES in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with the chembl data.

    Returns
    -------
    pd.DataFrame
        A dataframe with the chembl data and cleaned SMILES.
    """
    df["smiles"] = [smiles2smiles(smiles) for smiles in df["smiles_raw"]]

    # Remove smiles that are `None`
    num_invalid = df["smiles"].isna().sum()
    logger.info(f"Deleted {num_invalid} ({num_invalid/len(df):.2%}) invalid smiles")
    df = df.dropna(subset="smiles")
    return df


def main() -> None:
    """Clean the chembl data."""
    chembl_df = pd.read_csv("data/assets/molecules/chembl.tab", sep="\t")
    chembl_df = chembl_df.rename({"smiles": "smiles_raw"}, axis=1)
    chembl_df = clean(chembl_df)
    out_file = "data/assets/molecules/chembl-sanitized.tab"
    chembl_df.to_csv(out_file, sep="\t", index=False)
    logger.info(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
