import sys
import logging
import pandas as pd
import winsound

import globals

#%% Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DATA_DIR = "./data"
files = [
    "17842264490753980280_shift-0.5_agents-16.arrow",
    "16874621291821625949_shift-0.5_agents-24.arrow",
    "16007220157225280629_shift-0.5_agents-512.arrow",
]

MAX_PREFERENCE = 1
MIN_PREFERENCE = -1

SORT_BY = globals.SORT_BY + ["generation_id"]

for filename in files:
    METRIC_COLS = {
        "estimate",
        "min_proxy_weight",
        "max_proxy_weight",
        "average_proxy_weight",
        "median_proxy_weight",
    }

    #%% Read
    logger.info(f"Reading file {filename}")
    df_raw = pd.read_feather(f"{DATA_DIR}/raw/{filename}")
    df_raw.sort_values(by=SORT_BY, inplace=True)
    df_raw.index.set_names(["ID"], inplace=True)
    df_raw.reset_index(inplace=True)
    df_raw["total_agents"] = (
        df_raw["number_of_proxies"] + df_raw["number_of_delegators"]
    )

    #%% Error
    logger.info("Merging for error. . .")
    df_merged = pd.merge(
        df_raw,
        df_raw.query("coordination_mechanism == 'All Agents'"),
        on=list(set(df_raw.columns) - METRIC_COLS - {"coordination_mechanism", "ID"}),
        suffixes=("", "_all_agents"),
    )

    logger.info("Calculating error. . .")
    new_cols = set()
    df_merged["error"] = df_merged["estimate"] - df_merged["estimate_all_agents"]
    new_cols.add("error")
    df_merged["error_as_percent_of_space"] = df_merged["error"] / (
        MAX_PREFERENCE - MIN_PREFERENCE
    )
    new_cols.add("error_as_percent_of_space")
    df_merged["squared_error"] = df_merged["error"] ** 2
    new_cols.add("squared_error")
    df_merged["error_as_percent_of_space_squared"] = (
        df_merged["error_as_percent_of_space"] ** 2
    )
    new_cols.add("error_as_percent_of_space_squared")
    df_merged["error_as_percent_of_space_abs"] = (
        df_merged["error_as_percent_of_space"] ** 2
    )
    new_cols.add("error_as_percent_of_space_abs")

    df_raw = pd.merge(df_raw, df_merged[list(new_cols) + ["ID"]], on=["ID"])
    METRIC_COLS |= new_cols

    #%% Improvement
    logger.info("Merging for improvement. . .")
    df_merged = pd.merge(
        df_raw,
        df_raw.query("coordination_mechanism == 'Active Only'"),
        on=list(set(df_raw.columns) - METRIC_COLS - {"coordination_mechanism", "ID"}),
        suffixes=("", "_active_only"),
    )

    logger.info("Calculating improvement. . .")
    new_cols = set()
    df_merged["improvement"] = abs(df_merged["error_active_only"]) - abs(
        df_merged["error"]
    )
    new_cols.add("improvement")
    df_merged["improvement_as_percent_of_space"] = df_merged["improvement"] / (
        MAX_PREFERENCE - MIN_PREFERENCE
    )
    new_cols.add("improvement_as_percent_of_space")

    df_raw = pd.merge(df_raw, df_merged[list(new_cols) + ["ID"]], on=["ID"])
    METRIC_COLS |= new_cols

    #%% Shifted
    logger.info("Comparing against shifted. . .")
    df_merged = pd.merge(
        df_raw,
        df_raw.query("shifted == True"),
        on=list(set(df_raw.columns) - METRIC_COLS - {"shifted", "ID"}),
        suffixes=("", "/shifted"),
    )

    new_cols = set()
    for col in METRIC_COLS:
        df_merged[f"shifted_diff/{col}"] = (
            df_merged[f"{col}"] - df_merged[f"{col}/shifted"]
        )
        new_cols.add(f"shifted_diff/{col}")

    df_raw = pd.merge(df_raw, df_merged[list(new_cols) + ["ID"]], on=["ID"])
    METRIC_COLS |= new_cols

    #%% Save processed
    logger.info("Saving processed. . .")
    df_raw.to_feather(f"{DATA_DIR}/processed_{filename}")

    #%% Describe
    logger.info(f"Describing {len(df_raw)} rows. . .")
    df_raw.drop(columns=["ID", "generation_id"], inplace=True)
    df_described: pd.DataFrame = df_raw.groupby(
        by=list(set(df_raw.columns) - METRIC_COLS)
    ).describe()

    #%% Clean up described
    logger.info("Cleaning up. . .")
    df_described.reset_index(inplace=True)
    df_described.columns = df_described.columns.map("/".join)
    df_described.rename(
        columns={c: c[:-1] if c.endswith("/") else c for c in df_described.columns},
        inplace=True,
    )
    df_described.drop(
        columns=[c for c in df_described.columns if "/count" in c],
        inplace=True,
    )

    #%% Save described
    logger.info("Saving described. . .")
    df_described.to_feather(f"{DATA_DIR}/described_{filename}")

#%% Done
logger.info("Done!")
winsound.Beep(750, 750)
