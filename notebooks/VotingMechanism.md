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
    name: python3
---

```python pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Read in the data
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
%%capture --no-stderr
try:
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    drive.mount('/content/drive/')
    %cd /content/drive/MyDrive/Thesis Notebooks
except ImportError:
    %pwd
```

```python pycharm={"name": "#%%\n"}
filepath = "data/PES_21384000_rows_1-30_step2.feather"
df_original = pd.read_feather(filepath)
df_original.info(memory_usage="deep")
df_original.head()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We're going to want the rows averaged, so let's do that here. . .
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
criteria_columns = ["SystemEstimate", "SquaredError"]
# parameter_columns = list(set(df_original.columns) - set(criteria_columns))
# df = df_original.groupby(parameter_columns).mean().dropna(axis=0).reset_index()
df = df_original
```

```python pycharm={"name": "#%%\n"}
df.info(memory_usage="deep")
df.head()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
# Analysis
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
## First look
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's start by taking a look at some stats for each voting mechanism. Below we have plotted the boxen plots for each plot mechanism, as well as the mean of each mechanism represented as the dots inside the boxen plots.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
plot = sns.boxenplot(data=df,
                     x="VotingMechanism",
                     y="SquaredError")
sns.pointplot(data=df,
              x="VotingMechanism",
              y="SquaredError",
              color="k",
              ax=plot)

plot.set(ylim=(0, 1))
plt.xticks(rotation=90);
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Hmm. . . It seems simply averaging the estimates generally works best. This isn't too surprising since most of our distributions are symmetric about the truth. However, this might not remain true for the non-symmetric distributions. Let's take a look at that.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
## Symmetric vs Asymmetric
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
asymmetric_distros = [
    "Beta_3_.3",
    "Beta_4_1",
]
symmetric_distros = set(df["ProxyDistribution"].unique()) \
                    & set(df["InactiveDistribution"].unique()) \
                    - set(asymmetric_distros)
```

```python pycharm={"name": "#%%\n"}
target_rows = df[
    df["ProxyDistribution"].isin(asymmetric_distros)
    & df["InactiveDistribution"].isin(asymmetric_distros)
]

pd.options.mode.chained_assignment = None
target_rows["ProxyDistribution"] = target_rows[
    "ProxyDistribution"].cat.remove_unused_categories()
target_rows["InactiveDistribution"] = target_rows[
    "InactiveDistribution"].cat.remove_unused_categories()
pd.options.mode.chained_assignment = 'warn'

plot = sns.catplot(data=target_rows,
                   x="VotingMechanism", y="SquaredError",
                   row="ProxyDistribution", col="InactiveDistribution",
                   kind='boxen')
plot.set(ylim=(0, 1))
for ax in plot.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)

plot.fig.subplots_adjust(top=0.95)
plot.fig.suptitle("Asymmetric x Asymmetric")
```

```python pycharm={"name": "#%%\n"}
target_rows = df[
    df["ProxyDistribution"].isin(asymmetric_distros)
    & ~df["InactiveDistribution"].isin(asymmetric_distros)
]

pd.options.mode.chained_assignment = None
target_rows["ProxyDistribution"] = target_rows[
    "ProxyDistribution"].cat.remove_unused_categories()
target_rows["InactiveDistribution"] = target_rows[
    "InactiveDistribution"].cat.remove_unused_categories()
pd.options.mode.chained_assignment = 'warn'

plot = sns.catplot(data=target_rows,
                   x="VotingMechanism", y="SquaredError",
                   row="ProxyDistribution", col="InactiveDistribution",
                   kind='boxen')
plot.set(ylim=(0, 1))
for ax in plot.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)

plot.fig.subplots_adjust(top=0.95)
plot.fig.suptitle("Asymmetric x Symmetric")
```

```python pycharm={"name": "#%%\n"}
target_rows = df[
    ~df["ProxyDistribution"].isin(asymmetric_distros)
    & ~df["InactiveDistribution"].isin(asymmetric_distros)
]

pd.options.mode.chained_assignment = None
target_rows["ProxyDistribution"] = target_rows[
    "ProxyDistribution"].cat.remove_unused_categories()
target_rows["InactiveDistribution"] = target_rows[
    "InactiveDistribution"].cat.remove_unused_categories()
pd.options.mode.chained_assignment = 'warn'

plot = sns.catplot(data=target_rows,
                   x="VotingMechanism", y="SquaredError",
                   row="ProxyDistribution", col="InactiveDistribution",
                   kind='boxen')
plot.set(ylim=(0, 1))
for ax in plot.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)

plot.fig.subplots_adjust(top=0.95)
plot.fig.suptitle("Symmetric x Symmetric")
```
