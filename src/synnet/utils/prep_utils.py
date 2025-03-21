"""
This file contains various utils for data preparation and preprocessing.
"""

import logging

import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def split_data_into_Xy(
    *,
    steps: sparse.csc_matrix,
    states: sparse.csc_matrix,
    num_rxn: int,
    d_knn_emb: int,
) -> dict[str, dict[str, sparse.csc_matrix]]:
    """Split the featurized data into X,y-chunks for the {act,rt1,rxn,rt2}-networks.

    Args:
        num_rxn (int): Number of reactions in the dataset.
        out_dim (int): Size of the output feature vectors (used in kNN-search for rt1,rt2)
    """
    # Deduce dimensionality (TODO: find more elegant way)
    d_act_emb = 1  # {0,1,2,3}
    d_rxn_emb = 1  # {0, ..., number of reaction ids}
    d_emb = steps.shape[1] - d_act_emb - d_knn_emb - d_rxn_emb - d_knn_emb

    # Extract data for each network...
    data = dict()

    # ... action data
    # X: [z_state]
    # y: [action id] (int)
    X = states
    y = steps[:, 0]
    data["act"] = {"X": X, "y": y}

    # ... reaction data
    # X: [state, z_reactant_1]
    # y: [reaction_id] (int)
    # but: delete all steps where we *end* syntrees, as that will not be followed by a reaction
    actions = steps[:, 0].A  # (n,1) as array to allow boolean
    isActionEnd = (actions == 3).squeeze()  # (n,)
    states = states[~isActionEnd]
    steps = steps[~isActionEnd]
    X = sparse.hstack([states, steps[:, (2 * d_knn_emb + 2) :]])  # (n,4*4096)
    y = steps[:, d_knn_emb + 1]  # (n,1)
    data["rxn"] = {"X": X, "y": y}

    # ... reactant 2 data
    # X: [state,z_mol1,OneHotEnc(rxn_id)]
    # y: [z_mol2]
    # but: delete all steps where we *merge* syntrees, as in that case we already have reactant1+2
    actions = steps[:, 0].A  # (n',1) as array to allow boolean
    isActionMerge = (actions == 2).squeeze()  # (n',)
    steps = steps[~isActionMerge]
    states = states[~isActionMerge]
    z_mol1 = steps[:, (2 * d_knn_emb + 2) :]
    rxn_ids = steps[:, (1 + d_knn_emb)]
    z_rxn_id = OneHotEncoder().fit(np.arange(num_rxn)[:, None]).transform(rxn_ids.A)
    X = sparse.hstack((states, z_mol1, z_rxn_id))  # (n,3*4096+4096+91)
    y = steps[:, (2 + d_knn_emb) : (2 * d_knn_emb + 2)]
    data["rt2"] = {"X": X, "y": y}

    # ... reactant 1 data
    # X: [z_state]
    # y: [z'_reactant_1]
    # but: delete all steps where we expand syntrees, as in that case we already have a reactant1
    actions = steps[:, 0].A  # (n',1) as array to allow boolean
    isActionExpand = (actions == 1).squeeze()  # (n',)
    steps = steps[~isActionExpand]
    states = states[~isActionExpand]
    zprime_mol1 = steps[:, 1 : (d_knn_emb + 1)]

    # Order invariance:
    # If this step has a bi-molecular reaction, present the 2nd molecule to the rt1 network.
    zprime_mol2 = steps[:, 1 + d_knn_emb + 1 : -d_emb]
    has2ndReactant = np.asarray(zprime_mol2.sum(1) > 0).squeeze()
    X_with_reactant_2 = states[has2ndReactant]
    y_with_reactant_2 = zprime_mol2[has2ndReactant]
    data["rt1_augmented"] = {"X": X_with_reactant_2, "y": y_with_reactant_2}

    X = states
    y = zprime_mol1
    data["rt1"] = {"X": X, "y": y}
    return data
