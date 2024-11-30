import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):

    data["kfold"] = -1

    data = data.sample(frac=1).reset_index(drop = True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.income.values)):
        data.loc[v_, "kfold"] = f

    return data


if __name__ == "__main__":
    df = pd.read_csv("../input/adult.csv")
    df = create_folds(df)
    df.to_csv("../input/adult_folds.csv", index = False)