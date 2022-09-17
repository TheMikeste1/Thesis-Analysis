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

```python
import functools
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as stats
import seaborn as sns
from tqdm.notebook import tqdm as tqdm_notebook

from utils import *
```

```python
img_path = 'ratios'
```

## Read in the data

```python
# fmt: off
```

```python
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

```python
df_original = get_data(
    [
        "data/PES_21168000_rows_all_dists.feather",
    ]
)
df_original.info(memory_usage="deep")
df_original.head()
```

We could leave the dataframe as-is, but we want the general performance of each setup so we'll average the estimate (since each truth is the same) and error.

```python
criteria_columns = ["SystemEstimate", "SquaredError"]
parameter_columns = list(set(df_original.columns) - set(criteria_columns))
# df = df_original.groupby(parameter_columns).mean().dropna(axis=0).reset_index()
df = df_original
```

```python
should_delete_original = True
if should_delete_original and "df_original" in dir():
    del df_original
```

```python
df.info(memory_usage="deep")
df.head()
```

# Analysis


## First look

```python
# Drop all WeightlessAverageAll since it's potentially bad in this dataframe
df = df[df["VotingMechanism"] != "WeightlessAverageAll"]

# Read in the clean WeightlessAverageAll df
df_weightless = get_data(
    [
        "data/PES_1152000_rows_WeightlessOnly_All_Distros.feather",
    ]
)

# Concat together
df = pd.concat([df, df_weightless]).reset_index(drop=True)

# Continue as normal
disable_chain_warning()
df["VotingMechanism"] = df["VotingMechanism"].astype("category")
df["InactiveWeightingMechanism"] = df["InactiveWeightingMechanism"].astype("category")
enable_chain_warning()
```

```python
df.loc[df["VotingMechanism"] == "WeightlessAverageAll", "ProxyDistribution"].unique()
```

```python
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

```python
df_plot = df.copy()
disable_chain_warning()
df_plot["VotingMechanism"] = pd.Categorical(
    df_plot["VotingMechanism"],
    ordered=True,
    categories=average_mechanisms + ["WeightlessAverageAll"] + candidate_mechanisms,
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
plt.xticks(rotation=90);
```

```python
should_save = True
if should_save:
    save_eps(plot.get_figure(), dir_=f"img/{img_path}", name="voting_mechanisms_comparison.eps")
```

## Do any beat WeightlessAverageAll?

```python
rows = []
group_by = list(
    set(parameter_columns)
    - {"ProxyCount", "InactiveCount", "InactiveExtent", "ProxyExtent"}
)
test_column = "SquaredError"
```

```python
should_create_test_table = False
if should_create_test_table:
    groups = {
        (g,) if not isinstance(g, tuple) else g
        for g in df[df["VotingMechanism"] != "WeightlessAverageAll"]
        .groupby(by=group_by)
        .groups
    }
    bar = tqdm_notebook(groups) if is_notebook() else tqdm(groups)
    for group in bar:
        group_row = dict()
        # Get the rows for this group
        target_rows = np.ones(len(df)).astype(bool)
        for (value, col) in zip(group, group_by):
            group_row[col] = value
            target_rows &= df[col] == value
        target = df.loc[target_rows, test_column]
        if len(target) == 0:
            print(f"Group {group} has 0 rows, skipping. . .")
            continue

        # Compare against all other groups
        out_row = dict(group_row)
        # Get the rows for the other group
        other_rows = np.ones(len(df)).astype(bool)
        other_group = list(group)
        other_group[group_by.index("InactiveWeightingMechanism")] = "NoOp"
        other_group[group_by.index("VotingMechanism")] = "WeightlessAverageAll"
        for (value, col) in zip(other_group, group_by):
            out_row[f"{col}Other"] = value
            other_rows &= df[col] == value
        others = df.loc[other_rows, test_column]
        if len(others) == 0:
            print(f"Group {other_group} has 0 rows, skipping. . .")
            continue
        result_less = stats.mannwhitneyu(x=target, y=others, alternative="less")
        out_row.update(
            {
                "Statistic": result_less.statistic,
                "PValueLesser": result_less.pvalue,
            }
        )
        rows.append(out_row)
    test_table = pd.DataFrame(rows)
    test_table = test_table.sort_values(
        list(
            set(test_table.columns)
            - {"Statistic", "PValueGreater", "PValueEqual", "PValueLesser"}
        )
    ).reset_index(drop=True)
    test_table.to_csv("weightless_avg_all_test_table.csv")
```

```python
test_table = pd.read_csv("weightless_avg_all_test_table.csv")
```

```python
test_table = test_table[["VotingMechanism", "InactiveWeightingMechanism", "ProxyDistribution", "InactiveDistribution", "PValueLesser"]]
```

```python
alpha = 0.05
```

```python
lessers = test_table[(test_table["PValueLesser"] < alpha)].reset_index(drop=True)
print("Less than Others")
display(lessers)
```

It looks like those that perform better always have at least on asymmetrical distribution. Is this correct?

```python
asymmetric_distros = [
    "Beta_3_.3",
    "Beta_4_1",
    "Beta_.3_3",
    "Beta_1_4",
]
```

```python
original_len = len(lessers)
asymm_len = len(
        lessers[
            (lessers["ProxyDistribution"].isin(asymmetric_distros))
            | (lessers["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

Wow, it looks like ~97% of the lesser error distributions have at least one distribution that is asymmetrical! But is this actually a lot? How many instances actually have an asymmetrical distribution?

```python
original_len = len(df)
asymm_len = len(
        df[
            (df["ProxyDistribution"].isin(asymmetric_distros))
            | (df["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

```python
# Test to make sure the test table has a similar population to the main dataframe
original_len = len(test_table)
asymm_len = len(
        test_table[
            (test_table["ProxyDistribution"].isin(asymmetric_distros))
            | (test_table["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

Incredibly, the normal population only has ~75% of instances with at least one asymmetrical distribution! This might actually indicate the Proxy Vote System works better than averaging on asymmetrical error distributions!


Are there any asymmetrical distributions that are *not* one of the lessers?

```python
df_asymms = df.loc[
    (df["ProxyDistribution"].isin(asymmetric_distros))
    | (df["InactiveDistribution"].isin(asymmetric_distros)),
    ["ProxyDistribution", "InactiveDistribution"],
].drop_duplicates().reset_index(drop=True)
df_asymms
```

```python
lesser_asymms = lessers.loc[
    (lessers["ProxyDistribution"].isin(asymmetric_distros))
    | (lessers["InactiveDistribution"].isin(asymmetric_distros)),
    ["ProxyDistribution", "InactiveDistribution"],
].drop_duplicates().reset_index(drop=True)
lesser_asymms
```

```python
df_different = (
    (
        pd.concat([df_asymms, lesser_asymms])
        .drop_duplicates(keep=False)
        .reset_index(drop=True)
    )
    .sort_values(by=["ProxyDistribution", "InactiveDistribution"])
    .reset_index(drop=True)
)
df_different
```

All distributions on both sides are asymmetrical. Additionally, having a more severe the bias in the distribution, as is the case for the .3 betas, seems to result in it not working as well when both distributions are asymmetrical.


### The next natural question is how many have *both* distributions asymmetrical?

```python
original_len = len(lessers)
asymm_len = len(
        lessers[
            (lessers["ProxyDistribution"].isin(asymmetric_distros))
            & (lessers["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```



```python
original_len = len(df)
asymm_len = len(
        df[
            (df["ProxyDistribution"].isin(asymmetric_distros))
            & (df["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

```python
# Test to make sure the test table has a similar population to the main dataframe
original_len = len(test_table)
asymm_len = len(
        test_table[
            (test_table["ProxyDistribution"].isin(asymmetric_distros))
            & (test_table["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

Huh.. That's significantly less than the normal population. Not what I was expecting. What population of any asymmetrical distribution is that?

```python
original_len = len(lessers[
            (lessers["ProxyDistribution"].isin(asymmetric_distros))
            | (lessers["InactiveDistribution"].isin(asymmetric_distros))
        ])
asymm_len = len(
        lessers[
            (lessers["ProxyDistribution"].isin(asymmetric_distros))
            & (lessers["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

```python
original_len = len(df[
            (df["ProxyDistribution"].isin(asymmetric_distros))
            | (df["InactiveDistribution"].isin(asymmetric_distros))
        ])
asymm_len = len(
        df[
            (df["ProxyDistribution"].isin(asymmetric_distros))
            & (df["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

```python
# Test to make sure the test table has a similar population to the main dataframe
original_len = len(test_table[
            (test_table["ProxyDistribution"].isin(asymmetric_distros))
            | (test_table["InactiveDistribution"].isin(asymmetric_distros))
        ])
asymm_len = len(
        test_table[
            (test_table["ProxyDistribution"].isin(asymmetric_distros))
            & (test_table["InactiveDistribution"].isin(asymmetric_distros))
        ]
)

print(f"{asymm_len}/{original_len}; {asymm_len / original_len * 100:.2f}%")
```

That's very strange. I wonder why that is?


### Which of the lessers are both asymmetrical?

```python
lessers[
    (
        (lessers["ProxyDistribution"].isin(asymmetric_distros))
        & (lessers["InactiveDistribution"].isin(asymmetric_distros))
    )
]
```

There does not seem to be any clear pattern, though the Inactive distribution has a lot of Beta(0.3, 0.3).


### But which of those lessers are *not* included in the asymmetrical group?

```python
lessers[
    ~(
        (lessers["ProxyDistribution"].isin(asymmetric_distros))
        | (lessers["InactiveDistribution"].isin(asymmetric_distros))
    )
]
```

Interesting. Beta(4, 4) and Gaussian are very similar distributions, while Beta(0.3, 0.3) is basically the opposite. Instead of a bell curve, it curves up at the edges. All the proxy distributions are the "bell curve" distributions, while the inactives are the Beta(0.3, 0.3).


Also of note is the mechanisms for which this works are the "stronger" ones, consisting of Ranked Choice and Mean, as well as the better weighting mechanisms.



