"""Unit tests for the encoding module."""

import unittest

import numpy as np

from synnet.encoding.embedding import MorganFingerprintEmbedding


class TestMorganFingerprintEmbedding(unittest.TestCase):
    """Test the Morgan emeddings of SMILES strings."""

    valid_smiles = [
        "CC(C)(C)C(C(=O)O)n1cn[nH]c1=O",
        "C=CC1CN(C(=O)OC(C)(C)C)CCC1CCO",
        "COC(=O)c1coc(-c2cnn(C)c2)n1",
        "CC(C)(C)OC(=O)N1CCN(c2cc(N)cc(Br)c2)CC1",
        "CN(C)C(=O)N1CCCNCC1.C",
    ]

    def setUp(self) -> None:
        """Set up the test."""
        self.fp_embedding = MorganFingerprintEmbedding()

    def test_default_fp_embedding_single(self) -> None:
        """Test the embedding of a single SMILES string."""
        fp = self.fp_embedding.transform_smiles(self.valid_smiles[0])
        self.assertIsInstance(fp, np.ndarray)
        self.assertEqual(fp.dtype, bool)
        self.assertEqual(fp.shape, (4096,))

    def test_fp_on_invalid_smiles(self) -> None:
        """Test the embedding of invalid SMILES strings."""
        with self.assertRaises(ValueError):
            self.fp_embedding.transform_smiles("invalid_smiles")
