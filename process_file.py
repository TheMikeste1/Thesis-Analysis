import sys
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DATA_DIR = "./data"
filename = "shift_10_percent.arrow"
METRIC_COLS = {
    "estimate",
    "min_proxy_weight",
    "max_proxy_weight",
    "average_proxy_weight",
}

logger.info("Reading file")
df_raw = pd.read_feather(f"{DATA_DIR}/raw/{filename}")
df_raw["total_agents"] = df_raw["number_of_proxies"] + df_raw["number_of_delegates"]

logger.info("Merging. . .")
df_all_agents = df_raw[df_raw["coordination_mechanism"] == "All Agents"]
df_merged = pd.merge(
    df_raw,
    df_all_agents,
    on=list(set(df_raw.columns) - METRIC_COLS - {"coordination_mechanism"}),
    suffixes=("", "_all_agents"),
)
logger.info("Calculating error")
df_raw["error"] = df_merged["estimate"] - df_merged["estimate_all_agents"]
METRIC_COLS.add("error")
df_raw["squared_error"] = df_raw["error"] ** 2
METRIC_COLS.add("squared_error")

logger.info(f"Describing {len(df_raw)} rows. . .")
df_processed: pd.DataFrame = df_raw.groupby(
    by=list(set(df_raw.columns) - METRIC_COLS - {"generation_id"})
).describe()

logger.info("Cleaning up. . .")
df_processed.reset_index(inplace=True)
df_processed.columns = df_processed.columns.map("/".join)
df_processed.rename(
    columns={c: c[:-1] if c.endswith("/") else c for c in df_processed.columns},
    inplace=True,
)
df_processed.drop(
    columns=[
        c
        for c in df_processed.columns
        if "generation_id" in c or "estimate" in c or "/count" in c
    ],
    inplace=True,
)
df_processed.to_feather(f"{DATA_DIR}/processed_{filename}")

logger.info("Done!")
