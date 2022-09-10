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
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as stats
import seaborn as sns
from utils import *
```

```python pycharm={"name": "#%%\n"}
img_path = 'weighting_mechanisms'
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
# We'll need to remove NoOp, since it's only used for the
# WeightlessAverageAll voting mechanism
df_original = df
df = df[df["InactiveWeightingMechanism"] != "NoOp"]
disable_chain_warning()
df["InactiveWeightingMechanism"] = df[
    "InactiveWeightingMechanism"
].cat.remove_unused_categories()
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
plot = sns.boxenplot(data=df, x="InactiveWeightingMechanism", y="SquaredError")
color = "k"
plot.plot(
    "InactiveWeightingMechanism",
    "SquaredError",
    data=df.groupby(by=["InactiveWeightingMechanism"]).mean().reset_index(),
    color=color,
    label="Mean Error",
    linestyle="-",
)
plot.scatter(
    "InactiveWeightingMechanism",
    "SquaredError",
    data=df.groupby(by=["InactiveWeightingMechanism"]).mean().reset_index(),
    color=color,
    label="_mean_error",
)
plot.legend(loc="upper right")
plot.set(ylim=(0, 1 * 1.1))
plt.xticks(rotation=90);
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.get_figure(), "weighting_mechanisms_comparison.eps", dir_=f"img/{img_path}")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
These actually look really close. It looks like EqualWeight is likely the worst, and Closest *might* be the best, but let's use ANOVA to see if there's actually a difference.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_group = df.groupby(by=["InactiveWeightingMechanism"])
test_results = stats.f_oneway(
    *[df_group.get_group(group)["SquaredError"] for group in df_group.groups]
)
print(
    f"With EqualWeight-- S: {test_results.statistic:.2f}; P: {test_results.pvalue:.2e}"
)
# The others are so close I'm going to perform a test without EqualWeight
test_results = stats.f_oneway(
    *[
        df_group.get_group(group)["SquaredError"]
        for group in df_group.groups
        if group != "EqualWeight"
    ]
)
print(
    f"Without EqualWeight-- S: {test_results.statistic:.2f}; P: {test_results.pvalue:.2e}"
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
It would definitely appear there is at least once difference, even excluding EqualWeight.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
### Population tests
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
We already know they aren't, but let's start by checking if the distributions are normal.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    sns.set(font_scale=1.25)
    plot = sns.displot(
        data=df,
        x="SquaredError",
        col="InactiveWeightingMechanism",
        col_wrap=2,
        kind="kde",
    )
    sns.set(font_scale=1)
    save_eps(plot.fig, "weighting_mechanisms_error_distribution.eps", dir_=f"img/{img_path}")
    plt.close(plot.fig)
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    sns.set(font_scale=1.25)
    plot = sns.displot(
        data=df,
        x="SystemEstimate",
        col="InactiveWeightingMechanism",
        col_wrap=2,
        kind="kde",
    )
    sns.set(font_scale=1)
    save_eps(plot.fig, dir_=f"img/{img_path}", name="weighting_mechanisms_estimate_distribution.eps")
    plt.close(plot.fig)
```

```python pycharm={"name": "#%%\n"}
alpha = 0.05
check_normality_by_group(df, "SquaredError", ["InactiveWeightingMechanism"], alpha)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Definitely not normal. Let's continue with U-tests.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
alpha = 0.05
```

```python pycharm={"name": "#%%\n"}
test_table = perform_utests_against_others(
    df, "SquaredError", ["InactiveWeightingMechanism"]
)
print("Greater than Others")
display(test_table[(test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(test_table[(test_table["PValueEqual"] < alpha)])

print("Less than Others")
display(test_table[(test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Unsurprisingly, EqualWeight has a greater error than the general population. Let's continue with one-on-one tests.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
test_table = perform_utests_against_others_individually(
    df, "SquaredError", ["InactiveWeightingMechanism"]
)
```

```python pycharm={"name": "#%%\n"}
print("Greater than Others")
display(test_table[(test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(test_table[(test_table["PValueEqual"] < alpha)])

print("Less than Others")
display(test_table[(test_table["PValueLesser"] < alpha)])
```

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("weighting-mechanisms-p-values")
# Add all the mechanisms as nodes
for vm in df["InactiveWeightingMechanism"].unique():
    dot.node(vm)
# Create edges from the lessers to those they beat
lessers = test_table[(test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    dot.edge(
        row["InactiveWeightingMechanism"],
        row["InactiveWeightingMechanismOther"],
        label=label,
    )
dot.render(format="eps", directory=f"img/{img_path}/")
dot
```
