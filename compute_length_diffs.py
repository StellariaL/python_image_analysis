'''
Append per-timepoint differences for each length measurement in a lengths CSV.

For every embryo (grouped by date and embryo, sorted by time) and every
'l_*' column, computes the change from the previous timepoint and writes it
as 'l_[xxx]_diff'. The earliest timepoint of each embryo has no previous
timepoint, so its difference is left blank.

Usage:
python compute_length_diffs.py "D:\\Ruoheng_Li\\20260910-dispbead_ant\\lengths-0910+0911.csv"
python compute_length_diffs.py input.csv --output with_diffs.csv
'''

import argparse

import pandas as pd

GROUP_COLUMNS = ("date", "embryo")
TIME_COLUMN = "time"


def add_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [c for c in GROUP_COLUMNS if c in df.columns]
    length_columns = [
        c
        for c in df.columns
        if c.startswith("l_") and not c.endswith("_diff")
    ]
    if not length_columns:
        raise ValueError("No 'l_*' length columns found in input CSV")

    df = df.drop(columns=[f"{c}_diff" for c in length_columns], errors="ignore")
    df = df.sort_values(group_columns + [TIME_COLUMN]).reset_index(drop=True)

    diffs = df.groupby(group_columns, sort=False)[length_columns].diff()
    diffs.columns = [f"{c}_diff" for c in length_columns]
    return pd.concat([df, diffs], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append per-timepoint length differences to a lengths CSV.",
    )
    parser.add_argument("input", help="Path to the lengths CSV")
    parser.add_argument(
        "--output",
        help="Path to write the result (default: overwrite the input CSV)",
    )
    args = parser.parse_args()

    output_path = args.output or args.input
    result = add_diff_columns(pd.read_csv(args.input))
    result.to_csv(output_path, index=False)

    diff_columns = [c for c in result.columns if c.endswith("_diff")]
    print(f"Appended {', '.join(diff_columns)} to {output_path}")


if __name__ == "__main__":
    main()
