'''
Compute Euclidean distance between two named landmark points and append
the result as a column in lengths.csv.

Reads XY coordinates from data/raw_measurements.csv (columns: X, Y,
embryo, time, name, aligned_time). For each (embryo, time, aligned_time)
row that has both points, writes hypot(dx, dy) under the given column
name. If that column already exists in the output, it is replaced.

Usage:
python compute_lengths.py a_notochord node l_notochord
python compute_lengths.py a_boundary p_boundary l_embryo
python compute_lengths.py node aip node_to_a --measurements data/raw_measurements.csv
python compute_lengths.py a_notochord node l_notochord --output data/lengths.csv

Point names in data/raw_measurements.csv:
a_boundary, a_notochord, aip, node, p_boundary
'''

import argparse
import os

import numpy as np
import pandas as pd

MEASUREMENTS_CSV = os.path.join("data", "raw_measurements.csv")
OUTPUT_CSV = os.path.join("data", "lengths.csv")
KEY_COLUMNS = ("date","embryo", "time", "treatment")


def compute_distance(
    measurements_path: str,
    point_a: str,
    point_b: str,
    column_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(measurements_path)
    points = df[df["name"].isin([point_a, point_b])].copy()

    coords = points.pivot_table(
        index=list(KEY_COLUMNS),
        columns="name",
        values=["X", "Y"],
        aggfunc="first",
    )

    has_both = coords[("X", point_a)].notna() & coords[("X", point_b)].notna()
    coords = coords.loc[has_both]

    dx = coords[("X", point_a)] - coords[("X", point_b)]
    dy = coords[("Y", point_a)] - coords[("Y", point_b)]

    result = coords.index.to_frame(index=False)
    result[column_name] = np.hypot(dx, dy).values
    return result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def append_to_lengths(
    new_data: pd.DataFrame,
    output_path: str,
    column_name: str,
) -> pd.DataFrame:
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if column_name in existing.columns:
            existing = existing.drop(columns=[column_name])
        output = existing.merge(
            new_data,
            on=list(KEY_COLUMNS),
            how="outer",
        )
    else:
        output = new_data

    return output.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure distance between two named points and append to lengths.csv.",
    )
    parser.add_argument("point_a", help="Name of the first point")
    parser.add_argument("point_b", help="Name of the second point")
    parser.add_argument(
        "column_name",
        help="Column name for the distance in lengths.csv",
    )
    parser.add_argument(
        "--measurements",
        default=MEASUREMENTS_CSV,
        help=f"Path to measurements CSV (default: {MEASUREMENTS_CSV})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help=f"Path to output CSV (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    distances = compute_distance(
        args.measurements,
        args.point_a,
        args.point_b,
        args.column_name,
    )
    output = append_to_lengths(distances, args.output, args.column_name)
    output.to_csv(args.output, index=False)
    print(
        f"Appended {len(distances)} distances as '{args.column_name}' "
        f"to {args.output} ({len(output)} rows total)"
    )


if __name__ == "__main__":
    main()
