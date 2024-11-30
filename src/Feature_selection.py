import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np


if __name__ == "__main__":
    data = fetch_california_housing()
    X = data['data']
    col_names = data["feature_names"]
    y = data["target"]

    df = pd.DataFrame(X, columns = col_names)

    df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)

    print(df.corr())
