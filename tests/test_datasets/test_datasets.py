"""Unittest for the datasets module."""

import json
import unittest
from pathlib import Path

from synnet.datasets import SyntreeDataset
from synnet.utils.data_utils import SyntheticTree


class TestSyntreeDataset(unittest.TestCase):
    """Unit test for the SyntheticTree dataset."""

    def setUp(self) -> None:
        """Set up the test."""
        self.base_dir = Path(__file__).parent.absolute()
        self.syntree_file = self.base_dir / "assets/syntree-small-simple.json"

    def test_loading(self) -> None:
        """Test loading syntree from a file."""
        if not self.syntree_file.exists():
            self.skipTest("Syntree file not found.")
        with open(self.syntree_file, "rt", encoding="UTF-8") as f:
            syntree_dict = json.load(f)
        syntree = SyntheticTree.from_dict(syntree_dict)
        SyntreeDataset(dataset=[syntree])
