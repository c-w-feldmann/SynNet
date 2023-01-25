import json
from pathlib import Path

import pytest

from synnet.datasets import ActSyntreeDataset
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet


@pytest.fixture
def syntree_simplified():
    """Load a simplified syntree where SMILES are replaced with "n<int>"."""
    return SyntheticTree.from_dict(
        dict(json.loads(Path("tests/assets/syntree-small-simple.json").read_text()))
    )


def test_chop_syntree(syntree_simplified: SyntheticTree):
    assert isinstance(syntree_simplified, SyntheticTree)
    chopped = ActSyntreeDataset.chop_syntree(syntree_simplified)
    ref_chopped = [
    {'num_action': 0,  'target': 0,  'state': ('n8', None, None),  'reaction_id': 12,  'reactant_1': 'n1',  'reactant_2': 'n2'}, {'num_action': 1,  'target': 0,  'state': ('n8', 'n3', None),  'reaction_id': 47,  'reactant_1': 'n4',  'reactant_2': 'n5'}, {'num_action': 2,  'target': 2,  'state': ('n8', 'n6', 'n3'),  'reaction_id': 15,  'reactant_1': 'n6',  'reactant_2': 'n3'}, {'num_action': 3,  'target': 1,  'state': ('n8', 'n7', None),  'reaction_id': 49,  'reactant_1': 'n7',  'reactant_2': None}, {'num_action': 4,  'target': 3,  'state': ('n8', 'n8', None),  'reaction_id': None,  'reactant_1': None,  'reactant_2': None},
    ]  # fmt: skip
    assert chopped == ref_chopped
