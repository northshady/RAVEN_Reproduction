"""RADIal four-chip ADC decoding used by the reproduction dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ADCFrameProcessor:
    num_samples = 512
    num_chirps = 256
    receivers_per_chip = 4
    num_receivers = 16

    def __init__(self, calibration_path: str | Path) -> None:
        calibration_path = Path(calibration_path).expanduser().resolve()
        calibration = np.load(calibration_path, allow_pickle=True).item()
        required = {"Signal", "Azimuth_table", "Elevation_table", "H"}
        missing = sorted(required - set(calibration))
        if missing:
            raise ValueError(f"Calibration table is missing keys: {missing}")

    def _decode_chip(self, adc: np.ndarray) -> np.ndarray:
        complex_values = adc[0::2] + 1j * adc[1::2]
        return np.reshape(
            complex_values,
            (self.num_samples, self.receivers_per_chip, self.num_chirps),
            order="F",
        ).transpose(0, 2, 1)

    def run(self, adc0, adc1, adc2, adc3) -> np.ndarray:
        frames = [self._decode_chip(value) for value in (adc0, adc1, adc2, adc3)]
        # The released receiver order is chip3, chip0, chip1, chip2.
        return np.concatenate((frames[3], frames[0], frames[1], frames[2]), axis=2)
