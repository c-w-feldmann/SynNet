import json
from pathlib import Path

import pytest

from synnet.datasets import ActSyntreeDataset
from synnet.utils.data_utils import SyntheticTree

base_dir = Path(__file__).parent.absolute()


@pytest.fixture
def syntree_simplified() -> SyntheticTree:
    """Load a simplified syntree where SMILES are replaced with "n<int>"."""
    return SyntheticTree.from_dict(
        dict(
            json.load(
                open(
                    base_dir / "assets/syntree-small-simple.json", "r", encoding="utf-8"
                )
            )
        )
    )


def test_chop_syntree(syntree_simplified: SyntheticTree) -> None:
    assert isinstance(syntree_simplified, SyntheticTree)
    chopped = ActSyntreeDataset.chop_syntree(syntree_simplified)
    ref_chopped = [
        {"target": 0, "state": ("n8", None, None), "num_action": 0},
        {"target": 0, "state": ("n8", "n3", None), "num_action": 1},
        {"target": 2, "state": ("n8", "n6", "n3"), "num_action": 2},
        {"target": 1, "state": ("n8", "n7", None), "num_action": 3},
        {"target": 3, "state": ("n8", "n8", None), "num_action": 4},
    ]
    assert chopped == ref_chopped
