"""Unit tests for the models module."""

import unittest
from pathlib import Path

from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.mlp import MLP

CHECKPOINT_ICLR_DIR = Path("checkpoints/iclr")


class TestModels(unittest.TestCase):
    """Unit tests for models."""

    @unittest.skipIf(
        not CHECKPOINT_ICLR_DIR.exists(), "Checkpoint directory does not exist"
    )
    def test_can_load_iclr_checkpoints(self) -> None:
        """Test if we can load all ICLR checkpoints."""
        for model in "act rt1 rxn rt2".split():
            ckpt_file = find_best_model_ckpt(Path(CHECKPOINT_ICLR_DIR) / model)

            model = load_mlp_from_ckpt(ckpt_file)
            self.assertIsInstance(model, MLP)
