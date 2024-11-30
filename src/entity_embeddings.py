import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
import torch 
import torch.nn as nn
import torch.nn.functional as F

def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model
    for entity embeddings
    :params data: this is pandas dataframe
    :params catcols: list of categorical column names
    :return: compiled tf.keras model
    """

    # init list of inputs for embeddings
    inputs = []

    # init list of outputs for embeddings
    outputs = []

    for c in catcols:

        num_unique_values = int(data[c].nunique())

        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        inp = nn.Embedding(num_unique_values + 1, embed_dim)

        out = inp(torch.arange(num_unique_values + 1))

        out = F.dropout(out, 0.3)

        out = out.view(-1, embed_dim)

        inputs.append(inp)

        outputs.append(out)

    x = torch.cat(outputs, dim=1)
    x = nn.BatchNorm1d(x.shape[1])(x)

    x = torch.ReLU()(nn.Linear(x.shape[1], 300)(x))

    x = F.dropout(x, 0.3)
    x = nn.BatchNorm1d(x.shape[1])(x)

    x = nn.ReLU()(nn.Linear(x.shape[1], 300)(x))
    x = F.dropout(x, 0.3)
    x = nn.BatchNorm1d(x.shape[1])(x)

    y = nn.Softmax(dim=1)(nn.Linear(x.shape[1], 2)(x))

    model = nn.Sequential(*inputs, nn.Linear(x.shape[1], 2))

    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

def run(fold):
    df = pd.read_csv("../input/cat_data/train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)
    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_valid = df[df.kfold == fold].reset_index(drop = True)

    model = create_model(df, features)
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(xtrain,
            ytrain_cat,
            validation_data = (xvalid, yvalid_cat),
            verbose=1,
            batch_size=1024,
            epochs=3
            )
    valid_preds = model.predict(xvalid)[:, 1]

    print(metrics.roc_auc_score(yvalid, valid_preds))

    K.clear_session()

if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)

