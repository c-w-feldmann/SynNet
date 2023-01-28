import json
from pathlib import Path

import pytest

from synnet.datasets import SynTreeChopper
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet


@pytest.fixture
def syntree_simplified():
    """Load a simplified syntree where SMILES are replaced with chars A,B,C etc."""
    return SyntheticTree.from_dict(
        dict(json.loads(Path("tests/assets/syntree-small-simple.json").read_text()))
    )


def test_chop_syntree(syntree_simplified: SyntheticTree):
    assert isinstance(syntree_simplified, SyntheticTree)
    chopped = SynTreeChopper.chop_syntree(syntree_simplified)
    ref_chopped = [
    {'num_action': 0,  'target': 0,  'state': ('H', None, None),  'reaction_id': "r1",  'reactant_1': 'A',  'reactant_2': 'B'}, {'num_action': 1,  'target': 0,  'state': ('H', 'C', None),  'reaction_id': "r2",  'reactant_1': 'D',  'reactant_2': 'E'}, {'num_action': 2,  'target': 2,  'state': ('H', 'F', 'C'),  'reaction_id': "r3",  'reactant_1': 'F',  'reactant_2': 'C'}, {'num_action': 3,  'target': 1,  'state': ('H', 'G', None),  'reaction_id': "r4",  'reactant_1': 'G',  'reactant_2': None}, {'num_action': 4,  'target': 3,  'state': ('H', 'H', None),  'reaction_id': None,  'reactant_1': None,  'reactant_2': None},
    ]  # fmt: skip
    assert chopped == ref_chopped
