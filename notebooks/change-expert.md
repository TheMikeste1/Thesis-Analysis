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
df_processed = pd.read_feather(f"{DATA_DIR}/processed_3739165392236705654.arrow")
df_processed = df_processed[
    (df_processed["shifted"])
    & (
        (df_processed["coordination_mechanism"] == "Active Only")
        | (df_processed["coordination_mechanism"] == "Expert")
    )
]
df_processed["coordination_mechanism"].cat.remove_unused_categories(inplace=True)

df_described = pd.read_feather(f"{DATA_DIR}/described_3739165392236705654.arrow")
df_all_agents = df_described[
    (df_described["coordination_mechanism"] == "All Agents")
    & (df_described["shifted"])
]
df_all_agents["coordination_mechanism"].cat.remove_unused_categories(inplace=True)
df_active_only = df_described[
    (df_described["coordination_mechanism"] == "Active Only")
    & (df_described["shifted"])
]
df_active_only["coordination_mechanism"].cat.remove_unused_categories(inplace=True)
df_described = df_described[
    (df_described["coordination_mechanism"] == "Expert") & (df_described["shifted"])
]
df_described["coordination_mechanism"].cat.remove_unused_categories(inplace=True)

df_described.info(memory_usage=True)
df_described
```

# Analysis

```python
facet = sns.catplot(
    data=df_processed,
    x="distribution",
    y="error_as_percent_of_space_abs",
    col="voting_mechanism",
    hue="coordination_mechanism",
    kind="boxen",
    sharey= False
)
```

```python
facet = sns.catplot(
    data=df_processed,
    x="distribution",
    y="improvement",
    col="voting_mechanism",
    hue="coordination_mechanism",
    kind="boxen",
    sharey= False
)
```

```python
facet = sns.relplot(
    data=pd.concat([df_described, df_active_only]),
    x="number_of_delegators",
    y="squared_error/mean",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
    facet_kws={"sharey": False}
)
facet.fig.subplots_adjust(top=0.9)
facet.fig.suptitle("Preference Change - Expert");
```

```python
facet = sns.relplot(
    data=pd.concat([df_described, df_active_only]),
    x="number_of_delegators",
    y="squared_error/mean",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
)
facet.set(ylim=(0, 1e-1))
facet.fig.subplots_adjust(top=0.9)
facet.fig.suptitle("Preference Change - Expert");
```

```python
facet = sns.relplot(
    data=pd.concat([df_described, df_active_only]).query("number_of_delegators < 256"),
    x="number_of_delegators",
    y="squared_error/mean",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
    facet_kws={"sharey": False}
)
facet.fig.subplots_adjust(top=0.9)
facet.fig.suptitle("Preference Change - Expert");
```
