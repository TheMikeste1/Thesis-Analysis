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
from utils import *
```

```python pycharm={"name": "#%%\n"}
img_path = 'combinations'
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
df = df_original.groupby(parameter_columns).mean().dropna(axis=0).reset_index()
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

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's start by taking a look at how each mechanism performs.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# We'll need to remove those removed in the individual mechanism notebooks
df_original = df
df = df[
    (df["InactiveWeightingMechanism"] != "NoOp")
    & (df["VotingMechanism"] != "WeightlessAverageAll")
]
disable_chain_warning()
df["InactiveWeightingMechanism"] = df[
    "InactiveWeightingMechanism"
].cat.remove_unused_categories()
df["VotingMechanism"] = df["VotingMechanism"].cat.remove_unused_categories()
enable_chain_warning()
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
weighting_mechanisms = ["Borda", "Closest", "Distance", "EqualWeight"]
```

```python pycharm={"name": "#%%\n"}
def custom_plot(data: pd.DataFrame, x, y, mean_var=None, x_plot_color="k", **kwargs):
    del kwargs["color"]
    plot = sns.boxenplot(data=data, x=x, y=y, **kwargs)

    if mean_var is not None:
        var = data[mean_var].unique()[0]
        mean = data.loc[data[mean_var] == var, y].mean()
        plot.plot(
            range(-1, len(data[x].unique()) + 2),
            [mean for _ in range(-1, len(data[x].unique()) + 2)],
            color="r",
            label=f"{mean_var} Mean {y}",
            linestyle="-.",
        )
    sns.pointplot(
        data=data.groupby(by=[x]).mean().reset_index(),
        x=x,
        y=y,
        color=x_plot_color,
        ax=plot,
    )
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()
disable_chain_warning()
df_plot["VotingMechanism"] = pd.Categorical(
    df_plot["VotingMechanism"],
    ordered=True,
    categories=average_mechanisms + [""] + candidate_mechanisms,
)
enable_chain_warning()


plot = sns.FacetGrid(
    data=df_plot,
    col="InactiveWeightingMechanism",
    col_wrap=2,
    sharey=True,
    ylim=(0, 1 * 1.1),
    height=4,
)

plotting_func = functools.partial(custom_plot, mean_var="InactiveWeightingMechanism")
plot.map_dataframe(plotting_func, x="VotingMechanism", y="SquaredError")

for axes in plot.axes.flat:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

del df_plot
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.fig, "combined_comparison.eps", dir_=f"img/{img_path}")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We can learn some interesting things from this graph. First, the candidate mechanisms are consistently higher than the average for each weighting mechanism. Secondly, the Runoff mechanisms are occasionally almost as bad as the candidate mechanisms!
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
We'll take a similar path as with voting mechanisms--let's start with population tests.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
### Population tests
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
group_cols = ["VotingMechanism", "InactiveWeightingMechanism"]
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We'll start with an ANOVA test to see if there's any reason to continue.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_group = df.groupby(by=group_cols)
stats.f_oneway(
    *[df_group.get_group(group)["SquaredError"] for group in df_group.groups]
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
There are definitely at least one difference between the mechanisms. Let's dive a little deeper.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
It doesn't look like any population is normal, but let's double check.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
alpha = 0.05
check_normality_by_group(df, "SquaredError", group_cols, alpha)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The data is definitely not normal. We'll use U-tests.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
alpha = 0.05
```

```python pycharm={"name": "#%%\n"}
test_table = perform_utests_against_others_individually(
    df, "SquaredError", group_cols
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Average Mechanisms
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_average = df[df["VotingMechanism"].isin(average_mechanisms)]
disable_chain_warning()
df_average["VotingMechanism"] = pd.Categorical(
    df_average["VotingMechanism"], ordered=True, categories=average_mechanisms
)
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
average_test_table = test_table[
    (test_table["VotingMechanism"].isin(average_mechanisms)) &
    (test_table["VotingMechanismOther"].isin(average_mechanisms))
]
```

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("all-combos-p-values")
# Add all the mechanisms as nodes
for vm, wm in df_average.groupby(group_cols).groups:
    dot.node(f"{vm}\n{wm}")
# Create edges from the lessers to those they beat
lessers = average_test_table[(average_test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    l_node = f'{row["VotingMechanism"]}\n{row["InactiveWeightingMechanism"]}'
    r_node = f'{row["VotingMechanismOther"]}\n{row["InactiveWeightingMechanismOther"]}'
    dot.edge(l_node, r_node, label=label)
dot.graph_attr["ratio"] = f"{9.5 / 11}"
dot.render(format="eps", directory=f"img/{img_path}/")
dot
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Candidate Mechanisms
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_candidate = df[df["VotingMechanism"].isin(candidate_mechanisms)]
disable_chain_warning()
df_candidate["VotingMechanism"] = pd.Categorical(
    df_candidate["VotingMechanism"], ordered=True, categories=candidate_mechanisms
)
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
candidate_test_table = test_table[
    (test_table["VotingMechanism"].isin(candidate_mechanisms)) &
    (test_table["VotingMechanismOther"].isin(candidate_mechanisms))
]
```

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("all-combos-p-values")
# Add all the mechanisms as nodes
for vm, wm in df_candidate.groupby(group_cols).groups:
    dot.node(f"{vm}\n{wm}")
# Create edges from the lessers to those they beat
lessers = candidate_test_table[(candidate_test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    l_node = f'{row["VotingMechanism"]}\n{row["InactiveWeightingMechanism"]}'
    r_node = f'{row["VotingMechanismOther"]}\n{row["InactiveWeightingMechanismOther"]}'
    dot.edge(l_node, r_node, label=label)
dot.graph_attr["ratio"] = f"{9.5 / 11}"
dot.render(format="eps", directory=f"img/{img_path}/")
dot
```
