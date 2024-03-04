"""data_inspection.py
Collection of functions to inspect data.
"""

from __future__ import annotations

import functools
from collections import Counter
from itertools import chain
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from synnet.utils.custom_types import PathType
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet


@functools.lru_cache(maxsize=10)
def load_syntree_collection_from_file(file: PathType) -> SyntheticTreeSet:
    return SyntheticTreeSet.load(file)


def count_chemicals(syntrees: Union[SyntheticTree, SyntheticTreeSet]) -> Counter[str]:
    """Extract chemicals (reactants+product) in syntrees."""
    if isinstance(syntrees, SyntheticTree):
        syntrees = SyntheticTreeSet([syntrees])
    cnt: Counter[str] = Counter()
    [
        cnt.update([chemical.smiles for chemical in st.chemicals])
        for st in syntrees.synthetic_tree_list
    ]
    return cnt


def count_building_blocks(
    syntrees: Union[SyntheticTree, SyntheticTreeSet]
) -> Counter[str]:
    """Extract chemicals, which are leafes, in syntrees."""
    if isinstance(syntrees, SyntheticTree):
        syntrees = SyntheticTreeSet([syntrees])
    cnt: Counter[str] = Counter()
    [cnt.update(st.leafs_as_smiles) for st in syntrees.synthetic_tree_list]
    return cnt


def count_depths(sts: SyntheticTreeSet, max_depth: int = 15) -> Counter[float]:
    """Count depths."""
    DEPTHS = range(1, max_depth + 1)
    depth_list = [st.depth for st in sts.synthetic_tree_list]
    cnt_dict = {float(k): 0 for k in DEPTHS}
    cnt_dict |= Counter(depth_list)
    return Counter(cnt_dict)


def _extract_action_ids(sts: SyntheticTreeSet) -> dict[int, list[int]]:
    """Extract the list of reaction ids for each syntree."""
    actions = {i: st.actions for i, st in enumerate(sts.synthetic_tree_list)}
    return actions


def count_actions(sts: SyntheticTreeSet) -> Counter[int]:
    """Count actions."""
    actions = _extract_action_ids(sts)

    action_ids = Counter(list(chain(*actions.values())))

    cnt_action_dict = dict.fromkeys(range(4), 0)
    cnt_action_dict.update(action_ids)
    return Counter(cnt_action_dict)


def count_num_actions(sts: SyntheticTreeSet) -> Counter[int]:
    """Count number of actions (better metric for "depth")"""
    return Counter([st.num_actions for st in sts.synthetic_tree_list])


def plot_num_actions(
    sts: SyntheticTreeSet,
    ax: Optional[plt.Axes] = None,
    **plt_kwargs: Any,
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


def _extract_reaction_ids(sts: SyntheticTreeSet) -> dict[int, list[int]]:
    """Extract the list of reaction ids for each syntree."""
    reactions = dict()

    for i, st in enumerate(sts.synthetic_tree_list):
        ids = [node.rxn_id for node in st.reactions]
        reactions[i] = ids

    return reactions


def count_reactions(sts: SyntheticTreeSet, nReactions: int = 91) -> Counter[int]:
    """Count the reaction ids."""
    reactions = _extract_reaction_ids(sts)

    reaction_ids = Counter(list(chain(*reactions.values())))

    cnt_rxn_ids = dict.fromkeys(range(nReactions), 0)
    cnt_rxn_ids.update(reaction_ids)
    cnt_rxn_ids = Counter(cnt_rxn_ids)
    return cnt_rxn_ids


def summarize_syntree_collection(sts: SyntheticTreeSet) -> dict[str, Any]:
    res = {
        "nTrees:": len(sts),
        "avg_num_actions": np.mean([st.num_actions for st in sts.synthetic_tree_list]),
        "counters": {
            "depths": count_depths(sts),
            "reactions": count_reactions(sts),
            "actions": count_actions(sts),
        },
    }
    return res


def reactions_used_less_than(data: dict[int, int], n: int) -> list[int]:
    return [i for i, count in data.items() if count < n]


def plot_reaction_heatmap(
    data: dict[int, int],
    n_reactions: int = 91,
    relative: bool = False,
    **kwargs: Any,
) -> plt.Axes:
    """Plot heatmap of reactions

    See:
      - https://stackoverflow.com/questions/63861760/add-text-on-plt-imshow
    """
    assert len(data) == n_reactions

    if relative:
        total = sum(data.values())
        plot_data = {k: v / total * 100 for k, v in data.items()}
    else:
        plot_data = {k: float(v) for k, v in data.items()}

    m, n = 10, 10
    assert len(plot_data) <= m * n
    mat = np.zeros((m * n), dtype=int)
    print(mat.shape)
    mat[: len(plot_data)] = list(plot_data.values())
    mat = mat.reshape((m, n))

    labels = np.array([str(i) if i < n_reactions else "-" for i in range(m * n)])
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


def cnt_to_dataframe(
    cnt: Counter[Any], index_name: Optional[str] = None
) -> pd.DataFrame:
    df = pd.DataFrame().from_dict(cnt, columns=["count"], orient="index")
    df["count_rel"] = df["count"] / df["count"].sum()
    df["count_rel"] = df["count_rel"].round(4)
    if index_name:
        df.index.name = index_name
    return df


def col_as_percentage(
    df: pd.DataFrame,
    cols: Optional[Union[list[str], str, pd.Index]] = None,
    replace: bool = False,
) -> pd.DataFrame:
    if not cols:
        cols = df.columns
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        new_col = f"{col}_rel" if not replace else col
        df[new_col] = df[col] / sum(df[col])
    return df
