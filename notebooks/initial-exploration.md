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
```

```python
DATA_DIR = "../data"
ID_COLS = { "generation_id", "shifted", "distribution", "number_of_proxies", "number_of_delegates" }
METRIC_COLS = {"estimate", "min_proxy_weight", "max_proxy_weight", "average_proxy_weight"}
```

```python
df_raw = None
if df_raw is None:
    df_raw = pd.read_feather(f"{DATA_DIR}/2023-02-12_20-35-29.arrow")
df_raw.info(memory_usage=True)
df_raw.describe()
df_raw
```

Let's start by grabbing the baselines and calculating the error.

```python
df_baselines = df_raw[df_raw["coordination_mechanism"].isin({"All Agents", "Active Only"})]
df_baselines
```
