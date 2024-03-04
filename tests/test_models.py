from pathlib import Path

import pytest

from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.mlp import MLP

CHECKPOINT_ICLR_DIR = "checkpoints/iclr"


@pytest.mark.skipif(
    not Path(
        CHECKPOINT_ICLR_DIR
    ).exists(),  # assume if path exits, then all 4 files exist
    reason="ICLR checkpoints are not available",
)
@pytest.mark.parametrize("model", "act rt1 rxn rt2".split())
def test_can_load_iclr_checkpoints(model: str) -> None:
    ckpt_file = find_best_model_ckpt(Path(CHECKPOINT_ICLR_DIR) / model)
    assert isinstance(ckpt_file, Path), f"Could not find checkpoint for {model}"

    model = load_mlp_from_ckpt(ckpt_file)
    assert isinstance(model, MLP)
