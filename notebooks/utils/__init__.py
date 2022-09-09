import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats as stats


def disable_chain_warning():
    pd.options.mode.chained_assignment = None


def enable_chain_warning():
    pd.options.mode.chained_assignment = "warn"


def get_data(filepaths: [str]) -> pd.DataFrame:
    return pd.concat([pd.read_feather(path) for path in filepaths]).reset_index(
        drop=True
    )


def perform_utests_against_others(
    df: pd.DataFrame, test_column, group_by: [str]
) -> pd.DataFrame:
    rows = []
    for group in df.groupby(by=group_by).groups:
        out_row = dict()
        if not isinstance(group, tuple):
            group = (group,)
        target_rows = np.ones(len(df)).astype(bool)
        for (value, col) in zip(group, group_by):
            out_row[col] = value
            target_rows &= df[col] == value

        target = df.loc[target_rows, test_column]
        others = df.loc[~target_rows, test_column]

        if len(target) == 0:
            print(f"Group {group} has 0 rows, skipping. . .")
            continue
        if len(others) == 0:
            print(f"No rows are not part of {group}, skipping. . .")
            continue

        result_greater = stats.mannwhitneyu(x=target, y=others, alternative="greater")
        result_not_equal = stats.mannwhitneyu(
            x=target, y=others, alternative="two-sided"
        )
        result_less = stats.mannwhitneyu(x=target, y=others, alternative="less")
        out_row.update(
            {
                "Statistic": result_greater.statistic,
                "PValueGreater": result_greater.pvalue,
                "PValueEqual": 1 - result_not_equal.pvalue,
                "PValueLesser": result_less.pvalue,
            }
        )
        rows.append(out_row)

    out = pd.DataFrame(rows)
    return out.sort_values(
        list(
            set(out.columns)
            - {"Statistic", "PValueGreater", "PValueEqual", "PValueLesser"}
        )
    ).reset_index(drop=True)


def perform_utests_against_others_individually(
    df: pd.DataFrame, test_column, group_by: [str]
) -> pd.DataFrame:
    rows = []
    groups = {
        (g,) if not isinstance(g, tuple) else g for g in df.groupby(by=group_by).groups
    }
    for group in groups:
        group_row = dict()
        # Get the rows for this group
        target_rows = np.ones(len(df)).astype(bool)
        for (value, col) in zip(group, group_by):
            group_row[col] = value
            target_rows &= df[col] == value
        target = df.loc[target_rows, test_column]
        if len(target) == 0:
            print(f"Group {group} has 0 rows, skipping. . .")
            continue

        # Compare against all other groups
        for other_group in groups - {group}:
            out_row = dict(group_row)
            # Get the rows for the other group
            other_rows = np.ones(len(df)).astype(bool)
            for (value, col) in zip(other_group, group_by):
                out_row[f"{col}Other"] = value
                other_rows &= df[col] == value
            others = df.loc[other_rows, test_column]
            if len(others) == 0:
                print(f"Group {other_group} has 0 rows, skipping. . .")
                continue
            result_greater = stats.mannwhitneyu(
                x=target, y=others, alternative="greater"
            )
            result_not_equal = stats.mannwhitneyu(
                x=target, y=others, alternative="two-sided"
            )
            result_less = stats.mannwhitneyu(x=target, y=others, alternative="less")
            out_row.update(
                {
                    "Statistic": result_greater.statistic,
                    "PValueGreater": result_greater.pvalue,
                    "PValueEqual": 1 - result_not_equal.pvalue,
                    "PValueLesser": result_less.pvalue,
                }
            )
            rows.append(out_row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        list(
            set(out.columns)
            - {"Statistic", "PValueGreater", "PValueEqual", "PValueLesser"}
        )
    ).reset_index(drop=True)


def save_eps(fig: plt.Figure, name: str, dir_: str = "img"):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig.savefig(f"{dir_}/{name}", format="eps", bbox_inches="tight")
