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

from globals import *
```

# Load Data

```python
DATA_DIR = "../data"

df_processed, df_described = load_data(DATA_DIR)
df_processed.query("discrete_vote == False", inplace=True)
df_described.query("discrete_vote == False", inplace=True)
```

Here we're focussed on when the expert mechanism with no preference change. Let's load specifically that data.

```python
df_processed["generation_id"].max()
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
    col="coordination_mechanism",
    hue="voting_mechanism",
    row="discrete_vote",
    kind="line",
)
```

```python
facet = sns.relplot(
    data=df_described.query("shifted == False"),
    x="number_of_delegators",
    y="shifted_diff/error_as_percent_of_space_abs/max",
    col="voting_mechanism",
    row="distribution",
    hue="coordination_mechanism",
    kind="line",
)
```
