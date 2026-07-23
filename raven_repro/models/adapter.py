"""Complex-ADC input adapter for the paper-equation RAVEN model."""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping

import torch
import torch.nn as nn

from .raven import PaperRAVEN, PaperRAVENConfig


_CONFIG_FIELDS = {field.name for field in fields(PaperRAVENConfig)}


class RAVENADC(nn.Module):
    """Map RADIal complex ADC batches to RAVEN real/imaginary input."""

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__()
        values = dict(config or {})
        unknown = sorted(set(values) - _CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported RAVEN configuration keys: {unknown}")
        model_config = PaperRAVENConfig(**values)
        if model_config.input_layout != "chirp_sample_channel":
            raise ValueError(
                "RAVENADC emits chirp_sample_channel input; configure that layout."
            )
        self.raven = PaperRAVEN(model_config)

    @staticmethod
    def adapt_complex_adc(adc: torch.Tensor) -> torch.Tensor:
        """Convert complex ``[B,16,512,256]`` to RI ``[B,256,512,32]``."""

        if adc.ndim != 4 or tuple(adc.shape[1:]) != (16, 512, 256):
            raise ValueError(
                "RADIal ADC batches must have shape [B,16,512,256], "
                f"got {tuple(adc.shape)}."
            )
        if not torch.is_complex(adc):
            raise TypeError("RAVENADC requires complex ADC input.")
        real = adc.real.permute(0, 3, 2, 1)
        imaginary = adc.imag.permute(0, 3, 2, 1)
        return torch.cat((real, imaginary), dim=-1).float().contiguous()

    @property
    def cfg(self) -> PaperRAVENConfig:
        return self.raven.cfg

    def logical_parameter_counts(self):
        return self.raven.logical_parameter_counts()

    def forward(self, adc: torch.Tensor):
        return self.raven(self.adapt_complex_adc(adc))
