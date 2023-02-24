---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

# Load Data

```python
DATA_DIR = "../data"
ID_COLS = { "shifted", "distribution", "number_of_proxies", "number_of_delegators", "total_agents", "distribution" }
MECHANISM_COLS = { "coordination_mechanism", "voting_mechanism" }
METRIC_COLS = {
    "error",
    "improvement",
    "improvement_as_percent_of_space",
    "error_as_percent_of_space",
    "error_as_percent_of_space_squared",
    "error_as_percent_of_space_abs",
    "estimate", "squared_error", "min_proxy_weight", "max_proxy_weight", "average_proxy_weight", "median_proxy_weight"
}
ADDITIONAL_METRICS = {"min", "max", "mean", "25%", "50%", "75%", "std"}
ALL_METRIC_COLS = {
    f"{metric}/{statistic}"
    for metric in METRIC_COLS
    for statistic in ADDITIONAL_METRICS
}
```

Here we're focussed on when the expert mechanism with no preference change. Let's load specifically that data.

```python
df_processed = pd.read_feather(f"{DATA_DIR}/processed_3739165392236705654.arrow")
df_described = pd.read_feather(f"{DATA_DIR}/described_3739165392236705654.arrow")
```

```python
df_described
```

# Analysis

```python
facet = sns.catplot(
    data=df_processed.query("shifted == False"),
    x="coordination_mechanism",
    y="error_as_percent_of_space_abs",
    col="voting_mechanism",
    row="distribution",
    kind="boxen",
    sharey= False
)
```

```python
facet = sns.catplot(
    data=df_processed.query("shifted == False"),
    x="coordination_mechanism",
    y="improvement",
    col="voting_mechanism",
    row="distribution",
    kind="boxen",
    sharey= False
)
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="number_of_delegators",
    y="improvement_as_percent_of_space/mean",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
)
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == True"),
    x="number_of_delegators",
    y="improvement_as_percent_of_space/mean",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
)
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == True"),
    x="number_of_delegators",
    y="shifted_diff/error_as_percent_of_space_abs/max",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
)
for ax in facet.axes.flat:
    plt.setp(ax.lines, alpha=.75)
```
