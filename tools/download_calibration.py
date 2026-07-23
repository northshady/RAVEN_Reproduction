#!/usr/bin/env python3
"""Download and verify Valeo's official RADIal calibration table."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = "https://raw.githubusercontent.com/valeoai/RADIal/main/SignalProcessing/CalibrationTable.npy"
SHA256 = "f920fff498e7e81fd182f6f6f66e25f66ce34087cdfe6f5514d75bb4546f7809"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/CalibrationTable.npy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        if digest(output) == SHA256:
            print(f"Calibration already verified: {output}")
            return
        raise RuntimeError(f"Existing calibration has the wrong checksum: {output}")
    temporary = output.with_suffix(output.suffix + ".download")
    urllib.request.urlretrieve(URL, temporary)
    actual = digest(temporary)
    if actual != SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Calibration checksum mismatch: expected {SHA256}, got {actual}")
    temporary.replace(output)
    print(f"Calibration downloaded and verified: {output}")


if __name__ == "__main__":
    main()
