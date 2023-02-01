"""data_inspection.py
Collection of functions to inspect data.
"""
import functools
from collections import Counter
from itertools import chain
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet


@functools.lru_cache(maxsize=10)
def load_syntree_collection_from_file(file):
    return SyntheticTreeSet.load(file)


def count_chemicals(syntrees: Union[SyntheticTree, SyntheticTreeSet]):
    """Extract chemicals (reactants+product) in syntrees."""
    if isinstance(syntrees, SyntheticTree):
        syntrees = [syntrees]
    cnt = Counter()
    [cnt.update(st.chemicals_as_smiles) for st in syntrees]
    return cnt


def count_building_blocks(syntrees: Union[SyntheticTree, SyntheticTreeSet]):
    """Extract chemicals, which are leafes, in syntrees."""
    if isinstance(syntrees, SyntheticTree):
        syntrees = [syntrees]
    cnt = Counter()
    [cnt.update(st.leafs_as_smiles) for st in syntrees]
    return cnt


def count_depths(sts: SyntheticTreeSet, max_depth: int = 15) -> Counter:
    """Count depths."""
    DEPTHS = range(1, max_depth + 1)
    cnt = {k: 0 for k in DEPTHS}
    depths = [st.depth for st in sts]
    cnt = {k: 0 for k in DEPTHS}
    cnt |= Counter(depths)
    return Counter(cnt)


def _extract_action_ids(sts: SyntheticTreeSet) -> Dict[int, List[int]]:
    """Extract the list of reaction ids for each syntree."""
    actions = {i: st.actions for i, st in enumerate(sts)}
    return actions


def count_actions(sts: SyntheticTreeSet) -> Counter:
    """Count actions."""
    actions = _extract_action_ids(sts)

    action_ids = Counter(list(chain(*actions.values())))

    cnt_action_ids = dict.fromkeys(range(4), 0)
    cnt_action_ids.update(action_ids)
    cnt_action_ids = Counter(cnt_action_ids)
    return cnt_action_ids


def count_num_actions(sts: SyntheticTreeSet) -> Counter:
    """Count number of actions (better metric for "depth")"""
    num_actions = Counter([st.num_actions for st in sts])
    return num_actions


def plot_num_actions(
    sts: SyntheticTreeSet, ax: Optional[plt.Axes] = None, **plt_kwargs
) -> plt.Axes:
    actions = count_num_actions(sts)
    # Plot actions (type `Counter`) as barplot:
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(actions.keys(), actions.values(), **plt_kwargs)
    ax.set(
        xlabel="Number of actions",
        ylabel="Number of syntrees",
        xticks=range(0, max(actions.keys())),
    )
    return ax


def _extract_reaction_ids(sts: SyntheticTreeSet) -> Dict[int, List[int]]:
    """Extract the list of reaction ids for each syntree."""
    reactions = dict()

    for i, st in enumerate(sts):
        ids = [node.rxn_id for node in st.reactions]
        reactions[i] = ids

    return reactions


def count_reactions(sts: SyntheticTreeSet, nReactions: int = 91) -> Counter:
    """Count the reaction ids."""
    reactions = _extract_reaction_ids(sts)

    reaction_ids = Counter(list(chain(*reactions.values())))

    cnt_rxn_ids = dict.fromkeys(range(nReactions), 0)
    cnt_rxn_ids.update(reaction_ids)
    cnt_rxn_ids = Counter(cnt_rxn_ids)
    return cnt_rxn_ids


def summarize_syntree_collection(sts: SyntheticTreeSet) -> dict:
    """Compute summary statistics for syntree collection."""
    #  dict(sorted(count_num_actions(syntree_coll).items()))
    res = {
        "metadata": sts.metadata,
        "nTrees:": len(sts),
        "avg_num_actions": np.mean([st.num_actions for st in sts]),
        "counters": {
            "depths": dict(sorted(count_depths(sts).items())),
            "reactions": dict(sorted(count_reactions(sts).items())),
            "actions": dict(sorted(count_actions(sts).items())),
            "num_actions": dict(sorted(count_num_actions(sts).items())),
        },
    }
    return res


def reactions_used_less_than(data: dict[int, int], n: int) -> List[int]:
    return [i for i, count in data.items() if count < n]


def plot_reaction_heatmap(
    data: dict[int, int], nReactions: int = 91, relative: bool = False, **kwargs
):
    """Plot heatmap of reactions

    See:
      - https://stackoverflow.com/questions/63861760/add-text-on-plt-imshow
    """
    assert len(data) == nReactions

    if relative:
        total = sum(data.values())
        data = {k: v / total * 100 for k, v in data.items()}

    m, n = 10, 10
    assert len(data) <= m * n
    mat = np.zeros((m * n), dtype=int)
    print(mat.shape)
    mat[: len(data)] = list(data.values())
    mat = mat.reshape((m, n))

    labels = [str(i) if i < nReactions else "-" for i in range(m * n)]
    labels = np.reshape(labels, (m, n))
    ax = sns.heatmap(
        mat,
        cmap="inferno",
        norm=None,
        annot=labels,
        annot_kws={"fontsize": 8},
        fmt="s",
        square=True,
        **kwargs,
    )
    ax.set(xticks=[], yticks=[])
    return ax


def cnt_to_dataframe(cnt: Counter, index_name: Optional[str] = None):
    df = pd.DataFrame().from_dict(cnt, columns=["count"], orient="index")
    df["count_rel"] = df["count"] / df["count"].sum()
    df["count_rel"] = df["count_rel"].round(4)
    if index_name:
        df.index.name = index_name
    return df


def col_as_percentage(df: pd.DataFrame, cols: List[str] = None, replace: bool = False):
    if not cols:
        cols = df.columns
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        new_col = f"{col}_rel" if not replace else col
        df[new_col] = df[col] / sum(df[col])
    return df
