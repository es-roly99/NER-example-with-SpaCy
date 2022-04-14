import pandas as pd


def read_file(path):
    try:
        return pd.read_csv(path)
    except ValueError as e:
        print(e)


class Dataset:

    def __init__(self, path=None):

        if path is not None:
            df = read_file(path)
            self.train = df[df['dataset'] == "train"]
            self.validation = df[df['dataset'] == "validation"]
            self.test = df[df['dataset'] == "test"]
        else:
            self.train = pd.DataFrame()
            self.validation = pd.DataFrame()
            self.test = pd.DataFrame()

