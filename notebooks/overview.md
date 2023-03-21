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
import pandas as pd
import seaborn as sns

from globals import *

sns.set(font_scale=1.15)
```

```python
import os
if not os.path.exists("./plots"):
    %mkdir "./plots"
del os
```

# Load Data

```python
DATA_DIR = "../data"
df_processed, df_described = load_data(DATA_DIR)
```

```python
set(df_described.columns)
```

# Analysis


## CM/VM

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="number_of_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    errorbar=None
)
facet.set(xlabel="Number of Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig("./plots/vm_col_cm_hue_error_as_percent_of_space_abs_mean.eps", format='eps')
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="number_of_delegators",
    y="error_as_percent_of_space_abs/std",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    errorbar=None
)
facet.set(xlabel="Number of Delegators", ylabel="| Error | as % of space std. deviation")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig("./plots/vm_col_cm_hue_error_as_percent_of_space_abs_std.eps", format='eps')
```

## Preference Change

```python
df_described["shifted_diff/abs_diff/error_as_percent_of_space_abs/mean_abs"] = abs(df_described["shifted_diff/abs_diff/error_as_percent_of_space_abs/mean"])
facet = sns.relplot(
    data=df_described,
    x="number_of_delegators",
    y="shifted_diff/abs_diff/error_as_percent_of_space_abs/mean_abs",
    hue="coordination_mechanism",
    col="voting_mechanism",
    kind="line",
    errorbar=None,
)
facet.set(xlabel="Number of Delegators", ylabel="Difference in error\nafter preference change")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}")
facet.savefig("./plots/abs_diff_from_preference_change_error_as_percent_of_space_abs_mean.eps", format='eps')
```

```python
facet = sns.relplot(
    data=df_described,
    x="number_of_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    row="shifted",
    kind="line",
    errorbar=None,
)
facet.set(xlabel="Number of Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}\nPreference change? {row_name}")
facet.savefig("./plots/preference_change_error_as_percent_of_space_abs_mean.eps", format='eps')
```

## Distribution

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="number_of_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    row="distribution",
    kind="line",
    errorbar=None,
)
facet.set(xlabel="Number of Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}\nPreference Distribution {row_name}")
facet.savefig("./plots/distribution_error_as_percent_of_space_abs_mean.eps", format='eps')
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="number_of_delegators",
    y="error_as_percent_of_space_abs/mean",
    hue="coordination_mechanism",
    col="voting_mechanism",
    row="distribution",
    kind="line",
    errorbar=None,
    facet_kws=dict(sharey=False)
)
facet.set(xlabel="Number of Delegators", ylabel="| Error | as % of space")
facet.legend.set_title("Coordination\nMechanism")
facet.set_titles("Voting Mechanism: {col_name}\nPreference Distribution {row_name}")
facet.savefig("./plots/distribution_different_scale_error_as_percent_of_space_abs_mean.eps", format='eps')
```
