import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils

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

        inp = layers.Input(shape=(1,))

        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=c
        )(inp)

        out = layers.SpatialDropout1D(0.3)(out)

        out = layers.Reshape(target_shape = (embed_dim,))(out)

        inputs.append(inp)

        outputs.append(out)

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation = "relu")(x)

    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=y)

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

