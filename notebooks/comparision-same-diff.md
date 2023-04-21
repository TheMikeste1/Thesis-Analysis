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
```

```python
filename = "5372052_shift-0.2_agents-24_weight_p1_c_0.2"
df_diff_described = pd.read_feather(f"{DATA_DIR}/described_{filename}.arrow")
df_diff_described.sort_values(by=SORT_BY, inplace=True)
df_diff_described["percent_delegators"] = 100 * df_diff_described["number_of_delegators"] / df_diff_described["total_agents"]
```

```python
filename = "5372052_shift-0.2_agents-24_weight_p1_c_1"
df_same_described = pd.read_feather(f"{DATA_DIR}/described_{filename}.arrow")
df_same_described.sort_values(by=SORT_BY, inplace=True)
df_same_described["percent_delegators"] = 100 * df_same_described["number_of_delegators"] / df_same_described["total_agents"]
```

```python
df_merged = pd.merge(
    df_same_described,
    df_diff_described,
    on=list(GENERATION_ID_COLS | X_COLS | MECHANISM_COLS | {"percent_delegators"}),
    suffixes=("_same", "_diff"),
)
```

```python
df_merged["difference_in_error_as_percent_of_space_abs/mean"] = df_merged["error_as_percent_of_space_abs/mean_diff"] - df_merged["error_as_percent_of_space_abs/mean_same"]
```

# Analysis

```python
facet = sns.relplot(
    data=df_same_described.query("shifted == False"),
    x="percent_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    col_order=["Median", "Mean", "Midrange"],
    errorbar=None
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")

facet = sns.relplot(
    data=df_diff_described.query("shifted == False"),
    x="percent_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    col_order=["Median", "Mean", "Midrange"],
    errorbar=None
)
facet.set(xlabel="%  Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
```

```python
facet = sns.relplot(
    data=df_merged.query("shifted == False"),
    x="percent_delegators",
    y="difference_in_error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    col_order=["Median", "Mean", "Midrange"],
    errorbar=None
)
facet.set(xlabel="%  Delegators", ylabel="Difference in | Preference | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig(f"{save_dir}/difference_abs_pref_percent_of_space.eps", format='eps')
```

```python
smoothing_factor = 0.75
df_merged["smoothed_difference_in_error_as_percent_of_space_abs/mean"] = df_merged["difference_in_error_as_percent_of_space_abs/mean"].ewm(alpha=(1 - smoothing_factor)).mean()
facet = sns.relplot(
    data=df_merged.query("shifted == False"),
    x="percent_delegators",
    y="smoothed_difference_in_error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    col_order=["Median", "Mean", "Midrange"],
    errorbar=None
)
facet.set(xlabel="%  Delegators", ylabel="Difference in | Preference | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig(f"{save_dir}/difference_abs_pref_percent_of_space_smoothed.eps", format='eps')
```

```python
facet = sns.lmplot(
    data=df_merged,
    x="percent_delegators",
    y="difference_in_error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    scatter=False,
    ci=None,
    col_order=["Median", "Mean", "Midrange"],
    order=4,
)
facet.set(xlabel="%  Delegators", ylabel="Difference in | Preference | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig(f"{save_dir}/difference_abs_pref_percent_of_space_fitted.eps", format='eps')
```
