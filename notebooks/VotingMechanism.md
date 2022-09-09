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
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as stats
import seaborn as sns
from utils import *
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Read in the data
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# fmt: off
```

```python pycharm={"name": "#%%\n"}
%%capture --no-stderr
try:
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    drive.mount('/content/drive/')
    %cd "/content/drive/MyDrive/Thesis Notebooks"
except ImportError:
    %pwd
# fmt: on
```

```python pycharm={"name": "#%%\n"}
df_original = get_data(
    [
        "data/PES_21168000_rows_all_dists.feather",
    ]
)
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
```

```python pycharm={"name": "#%%\n"}
df_plot = df.copy()
disable_chain_warning()
df_plot["VotingMechanism"] = pd.Categorical(
    df_plot["VotingMechanism"],
    ordered=True,
    categories=average_mechanisms + [""] + candidate_mechanisms,
)
enable_chain_warning()

plot = sns.boxenplot(data=df_plot, x="VotingMechanism", y="SquaredError")
color = "k"
plot.plot(
    "VotingMechanism",
    "SquaredError",
    data=df_plot.groupby(by=["VotingMechanism"]).mean().reset_index(),
    color=color,
    label="Mean Error",
    linestyle="-",
)
plot.scatter(
    "VotingMechanism",
    "SquaredError",
    data=df_plot.groupby(by=["VotingMechanism"]).mean().reset_index(),
    color=color,
    label="_mean_error",
)
del df_plot
plot.legend(loc="upper right")
plot.set(ylim=(0, 1 * 1.1))
plt.xticks(rotation=90)
```

```python pycharm={"name": "#%%\n"}
should_save = False
if should_save:
    save_eps(plot.get_figure(), "voting_mechanisms_comparison.eps")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As an aside, what would an even distribution look like with squared error?
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
y = np.arange(-1, 1, 0.001)
df_squared = pd.DataFrame(map(lambda x: x * x, y), columns=["y"])
df_squared["x"] = ""
df_test = pd.concat(
    [
        df_squared,
    ]
)
plot = sns.boxenplot(data=df_test, x="x", y="y")
plot.set(ylim=(-0 * 1.1, 1 * 1.1))
del df_test, df_squared, y
```

```python pycharm={"name": "#%%\n"}
should_save = False
if should_save:
    save_eps(plot.get_figure(), "expected_even_distribution_squared_error.eps")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
What about a gaussian distribution?
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
y = np.random.normal(0, 1 / 3, size=int(1 / 0.001))
df_squared = pd.DataFrame(map(lambda x: x * x, y), columns=["y"])
df_squared["x"] = ""
df_test = pd.concat(
    [
        df_squared,
    ]
)
plot = sns.boxenplot(data=df_test, x="x", y="y")
plot.set(ylim=(-0 * 1.1, 1 * 1.1))
del df_test, df_squared, y
```

```python pycharm={"name": "#%%\n"}
should_save = False
if should_save:
    save_eps(plot.get_figure(), "expected_gaussian_distribution_squared_error.eps")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Population tests
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
Anyway, the voting mechanism populations all look pretty close. Let's do some population tests to see if they're different from each other.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
We'll start with an ANOVA test to see if there's any reason to continue.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_group = df.groupby(by=["VotingMechanism"])
stats.f_oneway(
    *[df_group.get_group(group)["SquaredError"] for group in df_group.groups]
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
There are definitely at least one difference between the mechanisms. Let's dive a little deeper.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
It doesn't look like any population is normal, but let's double check.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
should_save = False
if should_save:
    sns.set(font_scale=1.25)
    plot = sns.displot(
        data=df_original,
        x="SquaredError",
        col="VotingMechanism",
        col_wrap=3,
        col_order=average_mechanisms + candidate_mechanisms + ["WeightlessAverageAll"],
        kind="kde",
    )
    sns.set(font_scale=1)
    save_eps(plot.fig, "voting_mechanisms_error_distribution.eps")
    plt.close(plot.fig)
```

```python pycharm={"name": "#%%\n"}
should_save = False
if should_save:
    sns.set(font_scale=1.25)
    plot = sns.displot(
        data=df_original,
        x="SystemEstimate",
        col="VotingMechanism",
        col_wrap=3,
        col_order=average_mechanisms + candidate_mechanisms + ["WeightlessAverageAll"],
        kind="kde",
    )
    sns.set(font_scale=1)
    save_eps(plot.fig, "voting_mechanisms_estimate_distribution.eps")
    plt.close(plot.fig)
```

```python pycharm={"name": "#%%\n"}
alpha = 0.05
print(f"P-score to be normal: {alpha}\n")
normal_mechs = set()
for target_mech in df["VotingMechanism"].unique():
    data = df[df["VotingMechanism"] == target_mech]
    population = data["SquaredError"]

    if stats.normaltest(population)[1] <= alpha:
        print(
            f"{target_mech} p-score: {stats.normaltest(population)[1]:.3f}; is likely NOT normal *****"
        )
    else:
        print(
            f"{target_mech} p-score: {stats.normaltest(population)[1]:.3f}; is likely normal"
        )
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
test_table = perform_utests_against_others(df, "SquaredError", ["VotingMechanism"])
print("Greater than Others")
display(test_table[(test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(test_table[(test_table["PValueEqual"] < alpha)])

print("Less than Others")
display(test_table[(test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Interestingly, the candidate mechanisms seem to perform worse than the average mechanisms. This isn't too surprising, since a single candidate probably isn't going to have the exactly correct estimate. However, let's confirm this.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
target = df[df["VotingMechanism"].isin(average_mechanisms)]
others = df[df["VotingMechanism"].isin(candidate_mechanisms)]
stats.mannwhitneyu(
    x=target["SquaredError"], y=others["SquaredError"], alternative="less"
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
There is indeed a difference. Let's take a quick peak at how each of them beat each other.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
test_table = perform_utests_against_others_individually(
    df, "SquaredError", ["VotingMechanism"]
)
```

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("all-voting-mechanisms-p-values")
# Add all the mechanisms as nodes
for vm in df["VotingMechanism"].unique():
    dot.node(vm)
# Create edges from the lessers to those they beat
lessers = test_table[(test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    dot.edge(row["VotingMechanism"], row["VotingMechanismOther"], label=label)
dot.graph_attr["ratio"] = "0.86363636363636363636363636363636"
dot.render(format="eps", directory="img")
dot
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Average Mechanisms
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_average = df[df["VotingMechanism"].isin(average_mechanisms)]
disable_chain_warning()
df_average["VotingMechanism"] = pd.Categorical(
    df_average["VotingMechanism"], ordered=True, categories=average_mechanisms
)
enable_chain_warning()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's start by checking if there is actually a difference between the average mechanisms.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_group = df_average.groupby(by="VotingMechanism")
anova_stat = stats.f_oneway(
    *[df_group.get_group(group)["SquaredError"] for group in df_group.groups]
)
del df_group
anova_stat
```

<!-- #region pycharm={"name": "#%% md\n"} -->
There is definitely a difference, even between these mechanisms! Let's compare them one-on-one.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
average_test_table = test_table[
    (test_table["VotingMechanism"].isin(average_mechanisms)) &
    (test_table["VotingMechanismOther"].isin(average_mechanisms))
]
print("Greater than Other")
display(average_test_table[(average_test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(average_test_table[(average_test_table["PValueEqual"] < alpha)])

print("Less than Other")
display(average_test_table[(average_test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's also create a simple graph to make this easier to visualize.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("average-mechanisms-p-values")
# Add all the mechanisms as nodes
for vm in df_average["VotingMechanism"].unique():
    dot.node(vm)
# Create edges from the lessers to those they beat
lessers = average_test_table[(average_test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    dot.edge(row["VotingMechanism"], row["VotingMechanismOther"], label=label)
dot.render(format="eps", directory="img")
dot
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Candidate Mechanisms
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_candidate = df[df["VotingMechanism"].isin(candidate_mechanisms)]
disable_chain_warning()
df_candidate["VotingMechanism"] = pd.Categorical(
    df_candidate["VotingMechanism"], ordered=True, categories=candidate_mechanisms
)
enable_chain_warning()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Again, let's start by checking if there is actually a difference between the candidate mechanisms.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_group = df_candidate.groupby(by="VotingMechanism")
anova_stat = stats.f_oneway(
    *[df_group.get_group(group)["SquaredError"] for group in df_group.groups]
)
del df_group
anova_stat
```

<!-- #region pycharm={"name": "#%% md\n"} -->
There's also a difference between candidate mechanisms! Let's follow the same patterns as with average mechanisms.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
candidate_test_table = test_table[
    (test_table["VotingMechanism"].isin(candidate_mechanisms)) &
    (test_table["VotingMechanismOther"].isin(candidate_mechanisms))
]
print("Greater than Other")
display(candidate_test_table[(candidate_test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(candidate_test_table[(candidate_test_table["PValueEqual"] < alpha)])

print("Less than Other")
display(candidate_test_table[(candidate_test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's create a graph again to make this easier to visualize.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("candidate-mechanisms-p-values")
# Add all the mechanisms as nodes
for vm in df_candidate["VotingMechanism"].unique():
    dot.node(vm)
# Create edges from the lessers to those they beat
lessers = candidate_test_table[(candidate_test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    dot.edge(row["VotingMechanism"], row["VotingMechanismOther"], label=label)
dot.render(format="eps", directory="img")
dot
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Symmetric vs Asymmetric
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_candidate = df[df["VotingMechanism"].isin(candidate_mechanisms)]
disable_chain_warning()
df_candidate["VotingMechanism"] = pd.Categorical(
    df_candidate["VotingMechanism"], ordered=True, categories=candidate_mechanisms
)
enable_chain_warning()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's start by checking if there is actually a difference between the candidate mechanisms.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
df_group = df_candidate.groupby(by="VotingMechanism")
anova_stat = stats.f_oneway(
    *[df_group.get_group(group)["SquaredError"] for group in df_group.groups]
)
del df_group
anova_stat
```

<!-- #region pycharm={"name": "#%% md\n"} -->
There is definitely a difference, even between these mechanisms! Let's compare them one-on-one.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
candidate_test_table = test_table[
    (test_table["VotingMechanism"].isin(candidate_mechanisms)) &
    (test_table["VotingMechanismOther"].isin(candidate_mechanisms))
]
print("Greater than Other")
display(candidate_test_table[(candidate_test_table["PValueGreater"] < alpha)])

print("Equal to Others")
display(candidate_test_table[(candidate_test_table["PValueEqual"] < alpha)])

print("Less than Other")
display(candidate_test_table[(candidate_test_table["PValueLesser"] < alpha)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's also create a simple graph to make this easier to visualize.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
dot = gv.Digraph("candidate-mechanisms-p-values")
# Add all the mechanisms as nodes
for vm in df_candidate["VotingMechanism"].unique():
    dot.node(vm)
# Create edges from the lessers to those they beat
lessers = candidate_test_table[(candidate_test_table["PValueLesser"] < alpha)]
for _, row in lessers.iterrows():
    p_value = row["PValueLesser"]
    label = f"{p_value: .2f}" if p_value == 0 else f"{p_value: .2e}"
    dot.edge(row["VotingMechanism"], row["VotingMechanismOther"], label=label)
dot.render(format="eps", directory="img")
dot
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
    df["InactiveDistribution"].unique()
) - set(asymmetric_distros)
```
