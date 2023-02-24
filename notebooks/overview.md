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
ID_COLS = { "shifted", "distribution", "number_of_proxies", "total_agents", "distribution" }
MECHANISM_COLS = { "coordination_mechanism", "voting_mechanism" }
METRIC_COLS = {"error", "squared_error", "min_proxy_weight", "max_proxy_weight", "average_proxy_weight", "median_proxy_weight"}
```

Here we're focussed on when the expert mechanism with no preference change. Let's load specifically that data.

```python
df_processed = pd.read_feather(f"{DATA_DIR}/processed_shift_10_percent.arrow")
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
