---
jupyter:
  jupytext:
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    name: python3
---

```python pycharm={"name": "#%%\n"}
import functools
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as stats
import seaborn as sns
from tqdm.notebook import tqdm as tqdm_notebook

from utils import *
```

```python pycharm={"name": "#%%\n"}
img_path = 'ratios'
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Read in the data
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# fmt: off
```

```python pycharm={"name": "#%%\n"}
%%capture --no-stderr
try:
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    drive.mount('/content/drive/')
    %cd "/content/drive/MyDrive/Thesis Notebooks"
except ImportError:
    %pwd
# fmt: on
```

```python pycharm={"name": "#%%\n"}
df_original = get_data(
    [
        "data/PES_21168000_rows_all_dists.feather",
    ]
)
df_original.info(memory_usage="deep")
df_original.head()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We could leave the dataframe as-is, but we want the general performance of each setup so we'll average the estimate (since each truth is the same) and error.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
criteria_columns = ["SystemEstimate", "SquaredError"]
parameter_columns = list(set(df_original.columns) - set(criteria_columns))
# df = df_original.groupby(parameter_columns).mean().dropna(axis=0).reset_index()
df = df_original
```

```python pycharm={"name": "#%%\n"}
should_delete_original = True
if should_delete_original and "df_original" in dir():
    del df_original
```

```python pycharm={"name": "#%%\n"}
df.info(memory_usage="deep")
df.head()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
# Analysis
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
## First look
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df = df[
    (df["VotingMechanism"] != "WeightlessAverageAll")
    | (df["InactiveWeightingMechanism"].isin({"Borda"}))
    | (df["ProxyDistribution"].isin({'Beta_.3_3', 'Beta_1_4'}))
    | (df["InactiveDistribution"].isin({'Beta_.3_3', 'Beta_1_4'}))
]
df.loc[
    (df["VotingMechanism"] == "WeightlessAverageAll"), "InactiveWeightingMechanism"
] = "NoOp"

# Clone all WeightlessAverageAll into each weighting mechanism so it appears on graphs as we want
for wm in df.loc[
    df["InactiveWeightingMechanism"] != "NoOp", "InactiveWeightingMechanism"
].unique():
    df_tmp = df[df["VotingMechanism"] == "WeightlessAverageAll"].copy()
    df_tmp["InactiveWeightingMechanism"] = wm
    df = pd.concat([df, df_tmp])
df.reset_index(drop=True, inplace=True)

# df = df[df["InactiveWeightingMechanism"] != "NoOp"]

disable_chain_warning()
df["VotingMechanism"] = df["VotingMechanism"].astype("category")
df["InactiveWeightingMechanism"] = df["InactiveWeightingMechanism"].astype("category")
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
df.loc[df["VotingMechanism"] == "WeightlessAverageAll", "ProxyDistribution"].unique()
```

```python pycharm={"name": "#%%\n"}
average_mechanisms = [
    "Mean",
    "InstantRunoffAverage",
    "WeightedInstantRunoffAverage",
    "RankedChoice",
]
candidate_mechanisms = [
    "Median",
    "InstantRunoffCandidate",
    "WeightedInstantRunoffCandidate",
    "Plurality",
]
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()
disable_chain_warning()
df_plot["VotingMechanism"] = pd.Categorical(
    df_plot["VotingMechanism"],
    ordered=True,
    categories=average_mechanisms + ["WeightlessAverageAll"] + candidate_mechanisms,
)
enable_chain_warning()

plot = sns.boxenplot(data=df_plot, x="VotingMechanism", y="SquaredError")
color = "k"
plot.plot(
    "VotingMechanism",
    "SquaredError",
    data=df_plot.groupby(by=["VotingMechanism"]).mean().reset_index(),
    color=color,
    label="Mean Error",
    linestyle="-",
)
plot.scatter(
    "VotingMechanism",
    "SquaredError",
    data=df_plot.groupby(by=["VotingMechanism"]).mean().reset_index(),
    color=color,
    label="_mean_error",
)
del df_plot
plot.legend(loc="upper right")
plot.set(ylim=(0, 1 * 1.1))
plt.xticks(rotation=90);
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.get_figure(), dir_=f"img/{img_path}", name="voting_mechanisms_comparison.eps")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Do any beat WeightlessAverageAll?
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
rows = []
group_by = list(
    set(parameter_columns)
    - {"ProxyCount", "InactiveCount", "InactiveExtent", "ProxyExtent"}
)
test_column = "SquaredError"
```

```python pycharm={"name": "#%%\n"}
# groups = {
#     (g,) if not isinstance(g, tuple) else g
#     for g in df[(df["VotingMechanism"].astype(str) + df["InactiveWeightingMechanism"].astype(str)).isin({
#         "RankedChoiceDistance",
#         "RankedChoiceClosest",
#         "RankedChoiceBorda",
#         "MeanClosest",
#         "MeanDistance",
#     })]
#     .groupby(by=group_by)
#     .groups
# }

groups = {
    (g,) if not isinstance(g, tuple) else g
    for g in df[df["VotingMechanism"] != "WeightlessAverageAll"]
    .groupby(by=group_by)
    .groups
}
```

```python pycharm={"name": "#%%\n"}
bar = tqdm_notebook(groups) if is_notebook() else tqdm(groups)
for group in bar:
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
    out_row = dict(group_row)
    # Get the rows for the other group
    other_rows = np.ones(len(df)).astype(bool)
    other_group = list(group)
    other_group[group_by.index("InactiveWeightingMechanism")] = "NoOp"
    other_group[group_by.index("VotingMechanism")] = "WeightlessAverageAll"
    for (value, col) in zip(other_group, group_by):
        out_row[f"{col}Other"] = value
        other_rows &= df[col] == value
    others = df.loc[other_rows, test_column]
    if len(others) == 0:
        print(f"Group {other_group} has 0 rows, skipping. . .")
        continue
    result_less = stats.mannwhitneyu(x=target, y=others, alternative="less")
    out_row.update(
        {
            "Statistic": result_less.statistic,
            "PValueLesser": result_less.pvalue,
        }
    )
    rows.append(out_row)
test_table = pd.DataFrame(rows)
test_table = test_table.sort_values(
    list(
        set(test_table.columns)
        - {"Statistic", "PValueGreater", "PValueEqual", "PValueLesser"}
    )
).reset_index(drop=True)
```

```python pycharm={"name": "#%%\n"}
test_table = test_table[["VotingMechanism", "InactiveWeightingMechanism", "ProxyDistribution", "InactiveDistribution", "PValueLesser"]]
```

```python pycharm={"name": "#%%\n"}
alpha = 0.05
```

```python pycharm={"name": "#%%\n"}
lessers = test_table[(test_table["PValueLesser"] < alpha)].reset_index(drop=True)
print("Less than Others")
display(lessers)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
It looks like those that perform better always have at least on asymmetrical distribution. Is this correct?
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
asymmetric_distros = [
    "Beta_3_.3",
    "Beta_4_1",
    "Beta_.3_3",
    "Beta_1_4",
]
```

```python pycharm={"name": "#%%\n"}
original_len = len(lessers)
asymm_len = len(
        lessers[
            (lessers["ProxyDistribution"].isin(asymmetric_distros))
            | (lessers["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len:.2f}%")
```

```python pycharm={"name": "#%%\n"}
original_len = len(test_table)
asymm_len = len(
        test_table[
            (test_table["ProxyDistribution"].isin(asymmetric_distros))
            | (test_table["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len:.2f}%")
```

```python pycharm={"name": "#%%\n"}
original_len = len(df)
asymm_len = len(
        df[
            (df["ProxyDistribution"].isin(asymmetric_distros))
            | (df["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len:.2f}%")
```
