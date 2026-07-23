from pathlib import Path

import numpy as np
import torch

from raven_repro.config import load_config
from raven_repro.evaluation.metrics import box_iou
from raven_repro.models import RAVENADC


ROOT = Path(__file__).resolve().parents[1]


def test_sequence_config_and_parameter_count():
    config = load_config(ROOT / "configs/raven_raw_adc_sequence.json")
    config["model"]["ssm_backend"] = "fallback"
    model = RAVENADC(config["model"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_427_460


def test_complex_adapter_layout():
    adc = torch.zeros((1, 16, 512, 256), dtype=torch.complex64)
    adc[:, 3] = 2 + 5j
    adapted = RAVENADC.adapt_complex_adc(adc)
    assert adapted.shape == (1, 256, 512, 32)
    assert torch.all(adapted[..., 3] == 2)
    assert torch.all(adapted[..., 19] == 5)


def test_identical_box_iou_is_one():
    box = np.asarray([1.0, 2.0, 4.0, 6.0], dtype=np.float32)
    assert box_iou(box, box[None])[0] == 1.0
