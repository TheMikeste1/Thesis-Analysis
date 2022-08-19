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
parameter_columns = list(set(df_original.columns) - set(criteria_columns))
df = df_original.groupby(parameter_columns)\
    .mean()\
    .dropna(axis=0)\
    .reset_index()
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
Let's start by taking a look at the average error for each voting mechanism.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df.groupby(by="VotingMechanism").mean()
```

```python pycharm={"name": "#%%\n"}
plot = sns.catplot(data=df.groupby(by="VotingMechanism", as_index=False).mean(),
                   x="VotingMechanism", y="SquaredError")
plot.set(ylim=(0, 1))
plt.xticks(rotation=90);
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Hmm. . . It seems simply averaging the estimates works best. This isn't too surprising since most of our distributions are symmetric about the truth. However, this might not remain true for the non-symmetric distributions.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
Before we take a look at that, let's also glance at the minimum and maximum for each mechanism as well.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
plot = sns.catplot(data=df.groupby(by="VotingMechanism", as_index=False).min(numeric_only=True),
                   x="VotingMechanism", y="SquaredError")
plot.set(ylim=(0, 1))
plt.xticks(rotation=90);
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The minimums for each look pretty close. Let's zoom in.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
plot = sns.catplot(data=df.groupby(by="VotingMechanism", as_index=False).min(numeric_only=True),
                   x="VotingMechanism", y="SquaredError")
plt.xticks(rotation=90);
```

```python pycharm={"name": "#%%\n"}
plot = sns.catplot(data=df.groupby(by="VotingMechanism", as_index=False).max(numeric_only=True),
                   x="VotingMechanism", y="SquaredError")
plot.set(ylim=(0, 1))
plt.xticks(rotation=90);
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The maximums look similar to the averages. Ultimately, based purely off the voting mechanism it seems WeightlessAverageAll is the best. Again, this might be because most distributions are symmetric about the truth. Let's take a closer look to what happens in symmetric vs asymmetric distributions.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
## Symmetric vs Asymmetric
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}

```
