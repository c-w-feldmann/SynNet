"""Clean chembl data.

Info:
    Some SMILES in the chembl data are invalid.
    We clean all SMILES by converting them from `smiles`-> `mol`->`smiles`.
"""
import pandas as pd
from functools import partial

from synnet.utils.custom_types import PathType

import datamol as dm


def load(file: PathType) -> pd.DataFrame:
    df = dm.read_csv(file, sep="\t")
    df = df.rename({"smiles": "smiles_raw"}, axis=1)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    func = partial(dm.sanitize_smiles, isomeric=False)
    smiles = dm.parallelized(func, df["smiles_raw"].to_list())

    df["smiles"] = smiles

    # Remove smiles that are `None`
    num_invalid = df["smiles"].isna().sum()
    print(f"Deleted {num_invalid} ({num_invalid/len(df):.2%}) invalid smiles")
    df = df.dropna(subset="smiles")
    return df


def to_file(df: pd.DataFrame, file: PathType) -> None:
    df.to_csv(file, sep="\t", index=False)
    print(f"Saved to: {file}")
    return None


if __name__ == "__main__":
    in_file = "data/assets/molecules/chembl.tab"
    chembl_df = load(in_file)
    chembl_df = clean(chembl_df)
    out_file = "data/assets/molecules/chembl-sanitized.tab"
    to_file(chembl_df, out_file)
