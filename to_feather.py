import os

import pandas as pd


def featherify_all_in(dir_with_files: str, verbose: bool = False):
    if not dir_with_files.endswith('/'):
        dir_with_files += '/'

    files = os.listdir(dir_with_files)
    if verbose:
        print(f"Checking {len(files)} files. . .")
    data_files = []
    for i, file in enumerate(files, start=1):
        if verbose and i % 100 == 0:
            print(
                f"Checking {i}/{len(files)} ({i / len(files) * 100:.2f}%). . .")
        path = f"{dir_with_files}{file}"
        if path.endswith(".csv") and os.path.isfile(path):
            data_files.append(path)

    if verbose:
        print(f"Featherify-ing {len(data_files)} files. . .")
    for i, file in enumerate(data_files, start=1):
        if verbose and i % 100 == 0:
            print(
                f"Featherify-ing {i}/{len(files)} ("
                f"{i / len(files) * 100:.2f}%). . .")
        featherify(file)


def featherify(path: str):
    assert path.endswith(".csv"), "Path must end with .csv"
    new_path = path.replace(".csv", ".feather")
    df = pd.read_csv(path)
    df.to_feather(new_path)
