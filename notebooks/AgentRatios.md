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

```python pycharm={"name": "#%%\n"}
df_original = df

df = df[
    (df["VotingMechanism"] != "WeightlessAverageAll")
    | (df["InactiveWeightingMechanism"] == "NoOp")
]

# Clone all WeightlessAverageAll into each weighting mechanism so it appears on graphs as we want
for wm in df.loc[
    df["InactiveWeightingMechanism"] != "NoOp", "InactiveWeightingMechanism"
].unique():
    df_tmp = df[df["VotingMechanism"] == "WeightlessAverageAll"].copy()
    df_tmp["InactiveWeightingMechanism"] = wm
    df = pd.concat([df, df_tmp])
df.reset_index(drop=True, inplace=True)

df = df[df["InactiveWeightingMechanism"] != "NoOp"]

disable_chain_warning()
df["VotingMechanism"] = df["VotingMechanism"].astype("category")
df["InactiveWeightingMechanism"] = df[
    "InactiveWeightingMechanism"
].astype("category")
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(projection='3d')

df_sample = df.sample(frac=0.0005)
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
for i, vm in enumerate(df_sample.groupby(by="InactiveWeightingMechanism").groups):
    values = df_sample[df_sample["InactiveWeightingMechanism"] == vm]
    ax.scatter(values["ProxyCount"], values["InactiveCount"], values["SquaredError"],
               marker=markers[i], label=vm)

ax.set_xlabel('ProxyCount')
ax.set_ylabel('InactiveCount')
ax.set_zlabel('SquaredError')
ax.legend();
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()

plot = sns.relplot(
    data=df_plot,
    x="ProxyCount",
    y="SquaredError",
    hue="VotingMechanism",
    col="InactiveWeightingMechanism",
    col_wrap=2,
    kind="line",
    errorbar=None,
)
plt.tight_layout()
plot.add_legend(loc="center right")
plot.figure.subplots_adjust(right=0.79)
del df_plot
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.fig, dir_=f"img/{img_path}", name="proxy_count.eps")
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()

plot = sns.relplot(
    data=df_plot,
    x="InactiveCount",
    y="SquaredError",
    hue="VotingMechanism",
    col="InactiveWeightingMechanism",
    col_wrap=2,
    kind="line",
    errorbar=None,
)
plt.tight_layout()
plot.add_legend(loc="center right")
plot.figure.subplots_adjust(right=0.79)
del df_plot
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.fig, dir_=f"img/{img_path}", name="inactive_count.eps")
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()
df_plot["Proxies:Inactive"] = df_plot["ProxyCount"] / df_plot["InactiveCount"]

plot = sns.relplot(
    data=df_plot[df_plot["Proxies:Inactive"] <= 5],
    x="Proxies:Inactive",
    y="SquaredError",
    hue="VotingMechanism",
    col="InactiveWeightingMechanism",
    col_wrap=2,
    kind="line",
    errorbar=None,
)
plt.tight_layout()
plot.add_legend(loc="center right")
plot.figure.subplots_adjust(right=0.79)
del df_plot
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.fig, dir_=f"img/{img_path}", name="ratios_zoomed.eps")
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()
df_plot["Proxies:Inactive"] = df_plot["ProxyCount"] / df_plot["InactiveCount"]

plot = sns.relplot(
    data=df_plot,
    x="Proxies:Inactive",
    y="SquaredError",
    hue="VotingMechanism",
    col="InactiveWeightingMechanism",
    col_wrap=2,
    kind="line",
    errorbar=None,
)
plt.tight_layout()
plot.add_legend(loc="center right")
plot.figure.subplots_adjust(right=0.79)
del df_plot
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    save_eps(plot.fig, dir_=f"img/{img_path}", name="ratios.eps")
```
