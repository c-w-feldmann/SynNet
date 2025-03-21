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
        model_list = ["act", "rt1", "rxn", "rt2"]
        for model_name in model_list:
            model_path = find_best_model_ckpt(Path(CHECKPOINT_ICLR_DIR) / model_name)
            model = load_mlp_from_ckpt(model_path)
            self.assertIsInstance(model, MLP)
