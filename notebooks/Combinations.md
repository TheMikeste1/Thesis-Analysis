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
    plot = sns.pointplot(
        data=data.groupby(by=[x]).mean().reset_index(),
        x=x,
        y=y,
        color=x_plot_color,
        # label=f"{x} Mean {y}",
        ax=plot,
    )
    # plot.plot(
    #     x,
    #     y,
    #     data=data.groupby(by=[x]).mean().reset_index(),
    #     color=x_plot_color,
    #     label=f"{x} Mean {y}",
    #     linestyle="-",
    # )
    # plot.scatter(
    #     x,
    #     y,
    #     data=data.groupby(by=[x]).mean().reset_index(),
    #     color=x_plot_color,
    #     label=f"_{x}_mean_{y}",
    # )
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
