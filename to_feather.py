import os

import pandas as pd


def featherify(dir_with_files: str, verbose: bool = False):
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
    dfs = dict()
    for i, file in enumerate(data_files, start=1):
        if verbose and i % 100 == 0:
            print(f"Reading {i}/{len(files)} ({i / len(files) * 100:.2f}%). . .")
        dfs[file] = pd.read_csv(file)
    if verbose:
        print("Done reading! Feathering. . .")
    for path, file in dfs.items():
        new_path = path.replace(".csv", ".feather")
        file.to_feather(new_path)


if __name__ == "__main__":
    featherify("./data/", verbose=True)
