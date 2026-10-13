"""
Extracts the TOLIMAN filter bandpass and the camera's mono QE from the source
spreadsheets into two-column CSVs (wavelength [nm], throughput [fraction 0-1]).

    python scripts/extract_throughput.py --bandpass "Bandpass-Sydney.xlsx" \
        --qe "ApproximateQE-BFS-U3-200S6.xlsx" --out data/throughput

The spreadsheets store wavelengths in nm and throughputs in %.
"""

import argparse
from pathlib import Path

import numpy as np
import openpyxl


def find_column(rows, name):
    """Returns the column index of the header cell matching `name` (case-insensitive)."""
    header = [str(cell).strip().lower() if cell is not None else "" for cell in rows[0]]
    matches = [j for j, cell in enumerate(header) if cell == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one '{name}' column, found {len(matches)}: {rows[0]}")
    return matches[0]


def extract(path, header, wavelength_offset):
    """
    Reads (wavelength, value) pairs. The wavelength column sits at `wavelength_offset`
    relative to the column titled `header` (the value column is the one after it).
    """
    rows = list(openpyxl.load_workbook(path, data_only=True).worksheets[0].iter_rows(values_only=True))
    j = find_column(rows, header)
    wl_col, val_col = j + wavelength_offset, j + wavelength_offset + 1
    pairs = [(r[wl_col], r[val_col]) for r in rows[1:] if r[wl_col] is not None and r[val_col] is not None]
    table = np.array(pairs, dtype=float)
    table[:, 1] /= 100  # percent -> fraction
    if np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f"{path}: wavelengths are not increasing")
    return table


def save(table, path, description):
    header = f"{description}\nwavelength_nm,throughput_fraction"
    np.savetxt(path, table, delimiter=",", header=header, fmt=["%.1f", "%.6f"])
    print(f"Wrote {path} ({len(table)} rows, {table[0, 0]:.0f}-{table[-1, 0]:.0f} nm)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bandpass", required=True)
    parser.add_argument("--qe", required=True)
    parser.add_argument("--out", default="data/throughput")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # "Bandpass" heads the wavelength column; its values are in the next column
    save(
        extract(args.bandpass, "Bandpass", 0),
        out / "bandpass.csv",
        f"TOLIMAN filter bandpass, 'Bandpass' columns of {Path(args.bandpass).name} (% -> fraction)",
    )
    # "Mono QE (%)" heads the value column; wavelengths are in the column before it
    save(
        extract(args.qe, "Mono QE (%)", -1),
        out / "qe_mono.csv",
        f"Camera mono QE, 'Mono QE (%)' column of {Path(args.qe).name} (% -> fraction)",
    )


if __name__ == "__main__":
    main()
