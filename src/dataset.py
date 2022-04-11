import pandas as pd


def read_file(path, separator):
    try:
        return pd.read_csv(path, sep=separator)
    except ValueError as e:
        print(e)


class Dataset:

    def __init__(self, path, separator):
        df = read_file(path, separator)
        self.train = df[df['dataset'] == "train"]
        self.test = df[df['dataset'] == "test"]
        self.validation = df[df['dataset'] == "validation"]
