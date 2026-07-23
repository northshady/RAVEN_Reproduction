#!/usr/bin/env python3
"""Build label-aligned RADIal ADC files from the official raw sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


EXPECTED_SHAPE = (512, 256, 16)
MANIFEST_NAME = ".adc_preprocessing.json"
PREPROCESSING = {
    "version": 3,
    "dc_removal": False,
    "amplitude_transform": "none",
    "output_shape": list(EXPECTED_SHAPE),
    "output_dtype": "complex128",
}
RADAR_KEYS = ("radar_ch0", "radar_ch1", "radar_ch2", "radar_ch3")


def read_manifest(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_manifest(path: Path, status: str, handled: int, requires_overwrite: bool) -> None:
    payload = {
        "status": status,
        "preprocessing": PREPROCESSING,
        "handled": handled,
        "requires_overwrite": requires_overwrite,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_adc(path: Path) -> bool:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    return value.shape == EXPECTED_SHAPE and np.iscomplexobj(value)


def save_adc(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration", default="assets/CalibrationTable.npy")
    parser.add_argument("--record", action="append")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Keep CLI help usable before optional raw-sequence dependencies are installed.
    from third_party.radial_dbreader import SyncReader
    from tools.adc_processing import ADCFrameProcessor

    raw_root = Path(args.raw_root).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    calibration = Path(args.calibration).expanduser().resolve()
    for path in (raw_root, labels_path, calibration):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    existing = list(output_dir.glob("adc_*.npy"))
    manifest = read_manifest(manifest_path)
    if existing and not args.overwrite:
        if manifest is None or manifest.get("preprocessing") != PREPROCESSING:
            raise RuntimeError("Existing ADC cache has a different/unknown preprocessing contract; use --overwrite.")
        if manifest.get("requires_overwrite"):
            raise RuntimeError("An interrupted overwrite must be resumed with --overwrite.")
    requires_overwrite = bool(args.overwrite and existing)
    write_manifest(manifest_path, "building", 0, requires_overwrite)

    labels = pd.read_csv(labels_path)
    required_columns = {"dataset", "index", "numSample"}
    if not required_columns.issubset(labels.columns):
        raise ValueError(f"labels.csv must contain {sorted(required_columns)}")
    records = sorted(str(value) for value in labels["dataset"].unique())
    if args.record:
        unknown = sorted(set(args.record) - set(records))
        if unknown:
            raise ValueError(f"Unknown records: {unknown}")
        records = [record for record in records if record in set(args.record)]

    processor = ADCFrameProcessor(calibration)
    handled = written = skipped = 0
    for record_number, record in enumerate(records, 1):
        rows = labels[labels.dataset == record]
        source_indices = sorted(int(value) for value in rows["index"].unique())
        print(f"[{record_number}/{len(records)}] {record}: {len(source_indices)} frames", flush=True)
        reader = SyncReader(str(raw_root / record), tolerance=20000, silent=True)
        for source_index in source_indices:
            sample_ids = rows[rows["index"] == source_index]["numSample"].unique()
            if len(sample_ids) != 1:
                raise RuntimeError(f"{record}/{source_index} maps to {sample_ids}")
            sample_id = int(sample_ids[0])
            output_path = output_dir / f"adc_{sample_id:06d}.npy"
            if output_path.exists() and not args.overwrite:
                if not valid_adc(output_path):
                    raise RuntimeError(f"Invalid existing ADC file: {output_path}")
                skipped += 1
            else:
                synchronized = reader.GetSensorData(source_index)
                missing = [key for key in RADAR_KEYS if key not in synchronized]
                if missing:
                    raise RuntimeError(f"{record}/{source_index} lacks {missing}")
                adc = processor.run(
                    *(synchronized[key]["data"] for key in RADAR_KEYS)
                )
                if adc.shape != EXPECTED_SHAPE or not np.iscomplexobj(adc):
                    raise RuntimeError(f"Unexpected ADC output: {adc.shape}/{adc.dtype}")
                save_adc(output_path, adc)
                written += 1
            handled += 1
            if handled % 100 == 0:
                print(f"handled={handled} written={written} skipped={skipped}", flush=True)
            if args.max_samples and handled >= args.max_samples:
                write_manifest(manifest_path, "building", handled, requires_overwrite)
                print(f"Stopped at max-samples={args.max_samples}")
                return
    write_manifest(manifest_path, "complete", handled, False)
    print(f"Complete: handled={handled}, written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()
