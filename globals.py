import pandas as pd

GENERATION_ID_COLS = {"shifted", "discrete_vote", "total_agents", "distribution"}
X_COLS = {
    "number_of_proxies",
    "number_of_delegators",
}
MECHANISM_COLS = {"coordination_mechanism", "voting_mechanism"}
METRIC_COLS = {
    "average_proxy_weight",
    "error",
    "error_as_percent_of_space",
    "error_as_percent_of_space_abs",
    "error_as_percent_of_space_squared",
    "estimate",
    "improvement",
    "improvement_as_percent_of_space",
    "max_proxy_weight",
    "median_proxy_weight",
    "min_proxy_weight",
    "squared_error",
}
METRIC_COLS |= {f"shifted_diff/{metric}" for metric in METRIC_COLS}
ADDITIONAL_METRICS = {"min", "max", "mean", "25%", "50%", "75%", "std"}
ALL_METRIC_COLS = {
    f"{metric}/{statistic}"
    for metric in METRIC_COLS
    for statistic in ADDITIONAL_METRICS
}

SORT_BY = [
    "coordination_mechanism",
    "voting_mechanism",
    "distribution",
    "shifted",
    "discrete_vote",
    "number_of_delegators",
]

missing_cols = set(SORT_BY) - X_COLS - GENERATION_ID_COLS - MECHANISM_COLS
assert not missing_cols, (
    "Sorted values should include all non-metric columns! " f"Missing {missing_cols}"
)
del missing_cols


def load_data(data_dir: str) -> (pd.DataFrame, pd.DataFrame):
    FILENAME = "2494359615335987012_shift-0.2_agents-24"
    df_processed = pd.read_feather(f"{data_dir}/processed_{FILENAME}.arrow")
    df_processed.sort_values(by=SORT_BY, inplace=True)

    df_described = pd.read_feather(f"{data_dir}/described_{FILENAME}.arrow")
    df_described.sort_values(by=SORT_BY, inplace=True)
    return df_processed, df_described
