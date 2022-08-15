---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python pycharm={"name": "#%%\n"}
import os
import itertools

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

```python pycharm={"name": "#%%\n"}
def get_dataframe_from_files(dir_with_files: str, verbose: bool = False) -> pd.DataFrame:
   if not dir_with_files.endswith('/'):
      dir_with_files += '/'

   files = os.listdir(dir_with_files)
   if verbose:
      print(f"Checking {len(files)} files. . .")
   data_files = []
   for i, file in enumerate(files, start=1):
      if verbose and i % 100 == 0:
         print(f"Checking {i}/{len(files)} ({i / len(files) * 100:.2f}%). . .")
      path = f"{dir_with_files}{file}"
      if path.endswith(".csv") and os.path.isfile(path):
         data_files.append(path)


   if verbose:
      print(f"Reading {len(data_files)} files. . .")
   dfs = []
   for i, file in enumerate(data_files, start=1):
      if verbose and i % 100 == 0:
         print(f"Reading {i}/{len(files)} ({i / len(files) * 100:.2f}%). . .")
      dfs.append(pd.read_csv(file))
   print("Done reading! Concatenating. . .")
   return pd.concat(dfs)
```

```python pycharm={"name": "#%%\n"}
df = get_dataframe_from_files("../data", True)
df.describe()
```

```python pycharm={"name": "#%%\n"}
df.head()
```

```python pycharm={"name": "#%%\n"}
non_error_columns = list(set(df.columns) - {"SquaredError", "SystemEstimate"})
df_averaged_error = df.groupby(non_error_columns).mean().reset_index()
```

```python pycharm={"name": "#%%\n"}
df_averaged_error
```

<!-- #region pycharm={"name": "#%% md\n"} -->

<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_averaged_error["proxy:inactive"] = df_averaged_error["ProxyCount"] / df_averaged_error["InactiveCount"]
```

```python pycharm={"name": "#%%\n"}
max_average_error = max(df_averaged_error["SquaredError"])
expert_dists = {"Uniform", "Gaussian", "Beta_4_4"}
```

<!-- #region pycharm={"name": "#%% md\n"} -->
# Weighting Mechanism
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's take a look at how the weighting mechanisms affect the squared error.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
## Proxy:Inactive
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
Pretty much all of these are linear, which is a little strange. They are also quite chaotic, which might indicate a problem.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
x_col = "proxy:inactive"
```

```python pycharm={"name": "#%%\n"}
# Expert to expert
target_rows = df_averaged_error[
   (df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
plot.set(xscale="log")
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
plot.set(xscale="log")
```

```python pycharm={"name": "#%%\n"}
# Expert to untrained
target_rows = df_averaged_error[
   (df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (~df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
plot.set(xscale="log")
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
plot.set(xscale="log")
```

```python pycharm={"name": "#%%\n"}
# Untrained to expert
target_rows = df_averaged_error[
   (~df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
plot.set(xscale="log")
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
plot.set(xscale="log")
```

```python pycharm={"name": "#%%\n"}
# Untrained to untrained
target_rows = df_averaged_error[
   (~df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (~df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
plot.set(xscale="log")
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
plot.set(xscale="log")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## InactiveCount
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
x_col = "InactiveCount"
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Gradual decrease as count goes up. Uniform:Uniform is worst, followed by Any:Uniform. Borda and Distance are usually very close, with Borda being slightly better. Closest is usually slightly worse than the other two.

Interestingly, Uniform:Any besides Uniform makes all mechanisms very close, probably because it lends itself to the expertise of the inactives.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Expert to expert
target_rows = df_averaged_error[
   (df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
While mostly linear, there is a slight increase in error as more untrained inactives are used. Even worse, the error is worse than using EqualWeight, which is the same as not even using the inactive agents to begin with!

The order of mechanisms is the same as before: Borda being the best, closely followed by Distance and a small gap to Closest. At best, Borda is slightly better than EqualWeight with Distance being about equivalent with distributions Uniform:$\beta(0.3, 0.3)$.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Expert to untrained
target_rows = df_averaged_error[
   (df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (~df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Surprisingly, the Closest mechanism seems to work best here, followed by Distance and then Borda. This is the exact opposite as the previous two setups! This setup also seems to work best with $\beta(0.3, 0.3)$, which is bimodal-symmetric. The more skewed the distribution of the proxies, the worse using expert inactives seems to work.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Untrained to expert
target_rows = df_averaged_error[
   (~df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Untrained to Untrained does not work very well. As expected, not having expert agents to rely on makes it hard for the system to estimate well. In some cases it's worse than EqualWeight. Generally, Borda or Closest are best, with Distance always being slightly worse than Borda. When worse than EqualWeight, adding more inactives makes it worse, while when better adding more makes slight increases.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Untrained to untrained
target_rows = df_averaged_error[
   (~df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (~df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## ProxyCount
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
x_col = "ProxyCount"
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Surprisingly, in Expert to Expert and Gaussian/$\beta(4, 4)$:Uniform it's generally best to have *fewer* proxies. It also seems you want fewer proxies when using the Closest mechanism. Borda and Distance, however, do seem to perform better with more proxies with the order of mechanisms remaining the same as in InactiveCount: Borda first, then Distance, and finally Closest.

As before, having both proxies and inactives being strong agents (Gaussian or $/beta(4, 4)$) is best.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Expert to expert
target_rows = df_averaged_error[
   (df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As with InactiveCount, Expert to Untrained does not work as well as expected. However, the difference is far more dramatic with increasing proxy counts. More proxies produces a significant increase in error, though the order of best mechanism remains the same. This might be because having more proxies gives the untrained inactive agents too many choices.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Expert to untrained
target_rows = df_averaged_error[
   (df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (~df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Relying on the expertise of inactive agents seems to work fairly well. This works best with strong experts, and is worse with skewed untrained. The Closest mechanism is actually generally best here.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Untrained to expert
target_rows = df_averaged_error[
   (~df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
More proxies generally seems to help a little in Untrained to Untrained, though the error is much higher than other setups.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Untrained to untrained
target_rows = df_averaged_error[
   (~df_averaged_error["ProxyDistribution"].isin(expert_dists)) &
   (~df_averaged_error["InactiveDistribution"].isin(expert_dists))
]
# All
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",)
# By distribution
plot = sns.relplot(x=x_col, y="SquaredError", kind="line",
                   data=target_rows, hue="InactiveWeightingMechanism",
                   col="InactiveDistribution", row="ProxyDistribution",)
```
