import json
from pathlib import Path

import pytest

from synnet.utils.datastructures import SyntheticTree, SyntheticTreeSet

SYNTREE_FILES = [
    "tests/assets/syntree-small-simple-1.json",
    "tests/assets/syntree-small-simple-2.json",
    "tests/assets/syntree-small-simple-3.json",
]


def syntree_from_json(file):
    return SyntheticTree.from_dict(dict(json.loads(Path(file).read_text())))


@pytest.fixture
def syntrees():
    return [syntree_from_json(f) for f in SYNTREE_FILES]


def test_can_create_syntree_collection(syntrees):
    syntree_collection = SyntheticTreeSet(syntrees)
    assert len(syntree_collection) == len(syntrees)


def test_can_save_syntree_collection(syntrees, tmp_path):
    _file = tmp_path / "syntree-collection.json.gz"
    syntree_collection = SyntheticTreeSet(syntrees)
    syntree_collection.save(_file)
    assert (_file).exists()
    assert len(list(tmp_path.iterdir())) == 1

    collection_from_file = SyntheticTreeSet.load(_file)
    assert len(collection_from_file) == len(syntrees)
    assert collection_from_file.from_file == str(_file.resolve())
