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
import pandas as pd
import seaborn as sns
```

# Load Data

```python
DATA_DIR = "../data"
ID_COLS = { "shifted", "distribution", "number_of_proxies", "total_agents", "distribution" }
MECHANISM_COLS = { "coordination_mechanism", "voting_mechanism" }
METRIC_COLS = {"error", "squared_error", "min_proxy_weight", "max_proxy_weight", "average_proxy_weight"}
```

```python
if "df_data" not in globals() or True:
    df_data = pd.read_feather(f"{DATA_DIR}/processed_shift_10_percent.arrow")
df_data.info(memory_usage=True)
df_data
```

# Analysis

```python
sns.relplot(
    data=df_data[
        (~df_data["shifted"])
        # & (~df_data["coordination_mechanism"].isin({"All Agents"}))
    ],
    x="number_of_delegates",
    y="error/mean",
    hue="coordination_mechanism",
    col="distribution",
    row="voting_mechanism",
    facet_kws=dict(sharey=False),
    kind="line"
);
```
