---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import seaborn as sns

from globals import *

sns.set(font_scale=1.15)
```

```python
import os

save_dir = "./plots/different_weight"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
```

# Load Data

```python
DATA_DIR = "../data"
df_processed, df_described = load_data(DATA_DIR)
df_described["percent_delegators"] = 100 * df_described["number_of_delegators"] / df_described["total_agents"]
```

```python
set(df_described.columns)
```

# Analysis


## CM/VM

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="percent_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    errorbar=None
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig(f"{save_dir}/vm_col_cm_hue_error_as_percent_of_space_abs_mean.eps", format='eps')
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="percent_delegators",
    y="error_as_percent_of_space_abs/std",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    errorbar=None
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space std. deviation")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig(f"{save_dir}/vm_col_cm_hue_error_as_percent_of_space_abs_std.eps", format='eps')
```

## Preference Change

```python
df_described["shifted_diff/abs_diff/error_as_percent_of_space_abs/mean_abs"] = abs(
    df_described["shifted_diff/abs_diff/error_as_percent_of_space_abs/mean"]
)

facet = sns.lmplot(
    data=df_described,
    x="percent_delegators",
    y="shifted_diff/abs_diff/error_as_percent_of_space_abs/mean_abs",
    hue="coordination_mechanism",
    col="voting_mechanism",
    scatter=False,
    ci=None,
    order=2,
)
facet.set(
    xlabel="%  Delegators", ylabel="Difference in error\nafter preference change"
)
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig(
    f"{save_dir}/abs_diff_from_preference_change_error_as_percent_of_space_abs_mean.eps",
    format="eps",
)
```

```python
facet = sns.relplot(
    data=df_described,
    x="percent_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    row="shifted",
    kind="line",
    errorbar=None,
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}\nPreference change? {row_name}")
facet.savefig(f"{save_dir}/preference_change_error_as_percent_of_space_abs_mean.eps", format='eps')
```

## Distribution

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="percent_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    row="distribution",
    kind="line",
    errorbar=None,
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}\nPreference Distribution: {row_name}")
facet.savefig(f"{save_dir}/distribution_error_as_percent_of_space_abs_mean.eps", format='eps')
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="percent_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    row="distribution",
    kind="line",
    errorbar=None,
    facet_kws=dict(sharey=False)
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}\nPreference Distribution: {row_name}")
facet.savefig(f"{save_dir}/distribution_different_scale_error_as_percent_of_space_abs_mean.eps", format='eps')
```

```python
if not os.path.exists(f"{save_dir}/distributions"):
    os.mkdir(f"{save_dir}/distributions")

dists = df_described["distribution"].unique()
max_y = max(df_described["error_as_percent_of_space_abs/mean"])
margin = max_y * 0.025

for d in dists:
    df_plot = df_described.query("distribution == @d")
    df_plot["distribution"] = df_plot["distribution"].cat.remove_unused_categories()
    facet = sns.relplot(
        data=df_plot.query("shifted == False"),
        x="percent_delegators",
        y="error_as_percent_of_space_abs/mean",
        hue="coordination_mechanism",
        col="voting_mechanism",
        row="distribution",
        kind="line",
        errorbar=None,
    )
    facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space", ylim=(-margin, max_y + margin))
    facet.legend.set_title("Coordination\nMechanism")
    facet.set_titles("Voting Mechanism: {col_name}\nPreference Distribution: {row_name}")
    facet.savefig(f"{save_dir}/distributions/{d}_error_as_percent_of_space_abs_mean.eps", format='eps')
```
