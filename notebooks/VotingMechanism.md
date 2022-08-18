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
import pandas as pd
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Read in the data
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
%%capture --no-stderr
try:
    from google.colab import drive
    drive.mount('/content/drive/')
    %cd /content/drive/MyDrive/Thesis Notebooks
except ImportError:
    %pwd
```

```python pycharm={"name": "#%%\n"}
filepath = "data/PES_21384000_rows_1-30_step2.feather"
df = pd.read_feather(filepath)
df.info(memory_usage="deep")
df.head()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
# Analysis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="o0h12eGHojYu" executionInfo={"status": "ok", "timestamp": 1660855342614, "user_tz": 240, "elapsed": 7, "user": {"displayName": "Michael Hegerhorst", "userId": "02834695619649644407"}} outputId="316df735-1a17-4cd8-b4d6-6c0531927287" pycharm={"name": "#%%\n"}
df = pd.read_feather("data/")
```
