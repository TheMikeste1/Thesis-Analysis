import sys
import logging
import pandas as pd
import winsound

#%% Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DATA_DIR = "./data"
filename = "3739165392236705654.arrow"
METRIC_COLS = {
    "estimate",
    "min_proxy_weight",
    "max_proxy_weight",
    "average_proxy_weight",
    "median_proxy_weight",
}
MAX_PREFERENCE = 1
MIN_PREFERENCE = -1

#%% Read
logger.info("Reading file")
df_raw = pd.read_feather(f"{DATA_DIR}/raw/{filename}")
df_raw["total_agents"] = df_raw["number_of_proxies"] + df_raw["number_of_delegators"]
df_raw["percent_delegators"] = df_raw["number_of_delegators"] / df_raw["total_agents"]

#%% Error
logger.info("Merging for error. . .")
df_merged = pd.merge(
    df_raw,
    df_raw[df_raw["coordination_mechanism"] == "All Agents"],
    on=list(set(df_raw.columns) - METRIC_COLS - {"coordination_mechanism"}),
    suffixes=("", "_all_agents"),
)
logger.info("Calculating error. . .")
df_raw["error"] = df_merged["estimate"] - df_merged["estimate_all_agents"]
METRIC_COLS.add("error")
df_raw["error_as_percent_of_space"] = df_raw["error"] / (
    MAX_PREFERENCE - MIN_PREFERENCE
)
METRIC_COLS.add("error_as_percent_of_space")
df_raw["squared_error"] = df_raw["error"] ** 2
METRIC_COLS.add("squared_error")
df_raw["error_as_percent_of_space_squared"] = df_raw["error_as_percent_of_space"] ** 2
METRIC_COLS.add("error_as_percent_of_space_squared")
df_raw["error_as_percent_of_space_abs"] = df_raw["error_as_percent_of_space"] ** 2
METRIC_COLS.add("error_as_percent_of_space_abs")

#%% Improvement
logger.info("Merging for improvement. . .")
df_merged = pd.merge(
    df_raw,
    df_raw[df_raw["coordination_mechanism"] == "Active Only"],
    on=list(set(df_raw.columns) - METRIC_COLS - {"coordination_mechanism"}),
    suffixes=("", "_active_only"),
)
logger.info("Calculating improvement. . .")
df_raw["improvement"] = abs(df_merged["error_active_only"]) - abs(df_merged["error"])
METRIC_COLS.add("improvement")
df_raw["improvement_as_percent_of_space"] = df_raw["improvement"] / (
    MAX_PREFERENCE - MIN_PREFERENCE
)
METRIC_COLS.add("improvement_as_percent_of_space")

#%% Shifted
logger.info("Comparing against shifted. . .")
df_merge = pd.merge(
    df_raw,
    df_raw.query("shifted == True"),
    on=list(set(df_raw.columns) - METRIC_COLS - {"shifted"}),
    suffixes=("", "/shifted"),
)
new_cols = []
for col in METRIC_COLS:
    df_raw[f"shifted_diff/{col}"] = df_merge[f"{col}"] - df_merge[f"{col}/shifted"]
    new_cols.append(f"shifted_diff/{col}")
METRIC_COLS.update(new_cols)

#%% Save processed
logger.info("Saving processed. . .")
df_raw.to_feather(f"{DATA_DIR}/processed_{filename}")

#%% Describe
logger.info(f"Describing {len(df_raw)} rows. . .")
df_described: pd.DataFrame = df_raw.groupby(
    by=list(set(df_raw.columns) - METRIC_COLS - {"generation_id"})
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
    columns=[c for c in df_described.columns if "generation_id" in c or "/count" in c],
    inplace=True,
)

#%% Save described
logger.info("Saving described. . .")
df_described.to_feather(f"{DATA_DIR}/described_{filename}")

#%% Done
logger.info("Done!")
winsound.Beep(500, 100)
