---
jupyter:
  jupytext:
    main_language: python
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
import numpy as np
import pandas as pd
from scipy import stats as stats
import seaborn as sns
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Functions
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
def disable_chain_warning():
    pd.options.mode.chained_assignment = None
```

```python pycharm={"name": "#%%\n"}
def enable_chain_warning():
    pd.options.mode.chained_assignment = "warn"
```

```python pycharm={"name": "#%%\n"}
def get_data(filepaths: [str]) -> pd.DataFrame:
    return pd.concat(
            [pd.read_feather(path) for path in filepaths]
    ).reset_index(drop=True)
```

```python pycharm={"name": "#%%\n"}
def perform_utests_against_others(df: pd.DataFrame, test_column,
                                  group_by: [str]) -> pd.DataFrame:
    rows = []
    for group in df.groupby(by=group_by).groups:
        out_row = dict()
        if not isinstance(group, tuple):
            group = (group,)
        target_rows = np.ones(len(df)).astype(bool)
        for (value, col) in zip(group, group_by):
            out_row[col] = value
            target_rows &= df[col] == value

        target = df.loc[target_rows, test_column]
        others = df.loc[~target_rows, test_column]

        result_greater = stats.mannwhitneyu(x=target, y=others,
                                            alternative="greater")
        result_not_equal = stats.mannwhitneyu(x=target, y=others,
                                              alternative="two-sided")
        result_less = stats.mannwhitneyu(x=target, y=others,
                                         alternative="less")
        out_row.update({
            "Statistic"    : result_greater.statistic,
            "PValueGreater": result_greater.pvalue,
            "PValueEqual"  : 1 - result_not_equal.pvalue,
            "PValueLesser" : result_less.pvalue})
        rows.append(out_row)

    out = pd.DataFrame(rows)
    return out\
        .sort_values(list(set(out.columns) - {"Statistic", "PValueGreater", "PValueEqual", "PValueLesser"}))\
        .reset_index(drop=True)
```

```python pycharm={"name": "#%%\n"}
def perform_utests_against_others_individually(df: pd.DataFrame, test_column,
                                               group_by: [str]) -> pd.DataFrame:
    rows = []
    groups = {
        (g,) if not isinstance(g, tuple) else g
        for g in df.groupby(by=group_by).groups
    }
    for group in groups:
        group_row = dict()
        # Get the rows for this group
        target_rows = np.ones(len(df)).astype(bool)
        for (value, col) in zip(group, group_by):
            group_row[f"{col}Target"] = value
            target_rows &= df[col] == value
        target = df.loc[target_rows, test_column]
        # Compare against all other groups
        for other_group in groups - {group}:
            out_row = dict(group_row)
            # Get the rows for the other group
            other_rows = np.ones(len(df)).astype(bool)
            for (value, col) in zip(other_group, group_by):
                out_row[f"{col}Other"] = value
                other_rows &= df[col] == value
            others = df.loc[other_rows, test_column]

            result_greater = stats.mannwhitneyu(x=target, y=others,
                                                alternative="greater")
            result_not_equal = stats.mannwhitneyu(x=target, y=others,
                                                  alternative="two-sided")
            result_less = stats.mannwhitneyu(x=target, y=others,
                                             alternative="less")
            out_row.update({
                "Statistic"    : result_greater.statistic,
                "PValueGreater": result_greater.pvalue,
                "PValueEqual"  : 1 - result_not_equal.pvalue,
                "PValueLesser" : result_less.pvalue})
            rows.append(out_row)
    out = pd.DataFrame(rows)
    return out\
        .sort_values(list(set(out.columns) - {"Statistic", "PValueGreater", "PValueEqual", "PValueLesser"}))\
        .reset_index(drop=True)
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
    %cd "/content/drive/MyDrive/Thesis Notebooks"
except ImportError:
    %pwd
```

```python pycharm={"name": "#%%\n"}
df_original = get_data([
    "data/PES_21168000_rows_all_dists.feather",
])
df_original.info(memory_usage="deep")
df_original.head()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We could leave the dataframe as-is, but we want the general performance of each setup so we'll average the estimate (since each truth is the same) and error.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
criteria_columns = ["SystemEstimate", "SquaredError"]
parameter_columns = list(set(df_original.columns) - set(criteria_columns))
df = df_original.groupby(parameter_columns).mean().dropna(axis=0).reset_index()
```

```python pycharm={"name": "#%%\n"}
should_delete_original = True
if should_delete_original and "df_original" in dir():
    del df_original
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
df_original = df
df = df[df["VotingMechanism"] != "WeightlessAverageAll"]
disable_chain_warning()
df["VotingMechanism"] = df["VotingMechanism"].cat.remove_unused_categories()
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
df.groupby(by=["VotingMechanism"]).mean().reset_index()
```

```python pycharm={"name": "#%%\n"}
average_mechanisms = [
    "Mean",
    "InstantRunoffAverage",
    "WeightedInstantRunoffAverage",
    "RankedChoice",
]
candidate_mechanisms = [
    "Median",
    "InstantRunoffCandidate",
    "WeightedInstantRunoffCandidate",
    "Plurality",
]
disable_chain_warning()
df["VotingMechanism"] = pd.Categorical(df["VotingMechanism"], ordered=True,
                                       categories=average_mechanisms + [""] + candidate_mechanisms)
enable_chain_warning()
```

```python pycharm={"name": "#%%\n"}
plot = sns.boxenplot(data=df,
                     x="VotingMechanism",
                     y="SquaredError")
color = "k"
plot.plot("VotingMechanism",
          "SquaredError",
          data=df.groupby(by=["VotingMechanism"]).mean().reset_index(),
          color=color,
          label="Mean Error",
          linestyle="-")
plot.scatter("VotingMechanism",
             "SquaredError",
             data=df.groupby(by=["VotingMechanism"]).mean().reset_index(),
             color=color,
             label="_mean_error")

plot.legend(loc="upper right")
plot.set(ylim=(0, 1 * 1.1))
plt.xticks(rotation=90);
```

```python pycharm={"name": "#%%\n"}
should_save = True
if should_save:
    plot.get_figure().savefig('voting_mechanisms_comparison.eps', format='eps')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Hmmm. . . Those all look pretty close. Let's do some population tests to see if they're different from each other.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
It doesn't look like any population is normal, but let's double check.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
alpha = 0.05
print(f"P-score to be normal: {alpha}\n")
normal_mechs = set()
for target_mech in df["VotingMechanism"].unique():
    data = df[df["VotingMechanism"] == target_mech]
    population = data["SquaredError"]

    if stats.normaltest(population)[1] <= alpha:
        print(
                f"{target_mech} p-score: {stats.normaltest(population)[1]:.3f}; is likely NOT normal *****")
    else:
        print(
                f"{target_mech} p-score: {stats.normaltest(population)[1]:.3f}; is likely normal")
        normal_mechs.add(target_mech)

print()
if normal_mechs:
    print(f"Normal Dists: {normal_mechs}")
else:
    print("No normal distributions")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The data is definitely not normal. We'll use U-tests.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
alpha = 0.05
```

```python pycharm={"name": "#%%\n"}
test_table = perform_utests_against_others(df, "SquaredError",
                                           ["VotingMechanism"])
```

```python pycharm={"name": "#%%\n"}
print("Greater than Others")
display(test_table[(test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(test_table[(test_table["PValueEqual"] < alpha)])

print("Less than Others")
display(test_table[(test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
!!!!!! `TODO: This statement is false`
Looks like all populations are about equal when compared to all others. What about when comparing the mechanism one-on-one?
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
raise Exception("The statement above is false and needs to be revised")
```

```python pycharm={"name": "#%%\n"}
test_table = perform_utests_against_others_individually(df, "SquaredError",
                                                        ["VotingMechanism"])
```

```python pycharm={"name": "#%%\n"}
print("Greater than Other")
display(test_table[(test_table["PValueGreater"] < alpha)])

print("Less than Other")
display(test_table[(test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Symmetric vs Asymmetric
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
asymmetric_distros = [
    "Beta_3_.3",
    "Beta_4_1",
    "Beta_.3_3",
    "Beta_1_4",
]
symmetric_distros = set(df["ProxyDistribution"].unique()) & set(
        df["InactiveDistribution"].unique()) - set(asymmetric_distros)
```

```python pycharm={"name": "#%%\n"}
target_rows = df[
    df["ProxyDistribution"].isin(asymmetric_distros)
    & df["InactiveDistribution"].isin(asymmetric_distros)
    ]

disable_chain_warning()
target_rows["ProxyDistribution"] = target_rows[
    "ProxyDistribution"].cat.remove_unused_categories()
target_rows["InactiveDistribution"] = target_rows[
    "InactiveDistribution"].cat.remove_unused_categories()
enable_chain_warning()

plot = sns.catplot(data=target_rows,
                   x="VotingMechanism", y="SquaredError",
                   row="ProxyDistribution", col="InactiveDistribution",
                   kind='boxen')
plot.set(ylim=(0, 1 * 1.1))
for ax in plot.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)

plot.fig.subplots_adjust(top=0.95)
plot.fig.suptitle("Asymmetric x Asymmetric")

rows = []
for (target_mech, proxy_dist, inactive_dist) in target_rows.groupby(
        by=["VotingMechanism", "ProxyDistribution",
            "InactiveDistribution"]).groups:
    target = target_rows.loc[
        target_rows["VotingMechanism"] == target_mech, "SquaredError"]
    others = target_rows.loc[
        (target_rows["ProxyDistribution"] == proxy_dist) &
        (target_rows["InactiveDistribution"] == inactive_dist),
        "SquaredError"]

    result_greater = stats.mannwhitneyu(x=target, y=others,
                                        alternative="greater")
    result_not_equal = stats.mannwhitneyu(x=target, y=others,
                                          alternative="two-sided")
    result_less = stats.mannwhitneyu(x=target, y=others, alternative="less")

    rows.append({
        "VotingMechanismTarget": target_mech,
        "ProxyDist"            : proxy_dist,
        "InactiveDist"         : inactive_dist,
        "PValueGreater"        : result_greater.pvalue,
        "PValueEqual"          : 1 - result_not_equal.pvalue,
        "PValueLesser"         : result_less.pvalue})

test_table = pd.DataFrame(rows)
test_table[test_table["PValueEqual"] < 1]
```

```python pycharm={"name": "#%%\n"}
test_table[test_table["PValueEqual"] < 1]
```

```python pycharm={"is_executing": true, "name": "#%%\n"}
target_rows = df[
    df["ProxyDistribution"].isin(asymmetric_distros)
    & ~df["InactiveDistribution"].isin(asymmetric_distros)
    ]

disable_chain_warning()
target_rows["ProxyDistribution"] = target_rows[
    "ProxyDistribution"].cat.remove_unused_categories()
target_rows["InactiveDistribution"] = target_rows[
    "InactiveDistribution"].cat.remove_unused_categories()
enable_chain_warning()

plot = sns.catplot(data=target_rows,
                   x="VotingMechanism", y="SquaredError",
                   row="ProxyDistribution", col="InactiveDistribution",
                   kind='boxen')
plot.set(ylim=(0, 1 * 1.1))
for ax in plot.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)

plot.fig.subplots_adjust(top=0.95)
plot.fig.suptitle("Asymmetric x Symmetric")
```

```python pycharm={"is_executing": true, "name": "#%%\n"}
target_rows = df[
    ~df["ProxyDistribution"].isin(asymmetric_distros)
    & ~df["InactiveDistribution"].isin(asymmetric_distros)
    ]

disable_chain_warning()
target_rows["ProxyDistribution"] = target_rows[
    "ProxyDistribution"].cat.remove_unused_categories()
target_rows["InactiveDistribution"] = target_rows[
    "InactiveDistribution"].cat.remove_unused_categories()
enable_chain_warning()

plot = sns.catplot(data=target_rows,
                   x="VotingMechanism", y="SquaredError",
                   row="ProxyDistribution", col="InactiveDistribution",
                   kind='boxen')
plot.set(ylim=(0, 1 * 1.1))
for ax in plot.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)

plot.fig.subplots_adjust(top=0.95)
plot.fig.suptitle("Symmetric x Symmetric")
```
