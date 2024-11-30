import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from PIL import Image
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm

def create_dataset(training_df, image_dir):
    """
    This function takes the training dataframe
    and outputs training array and labels
    :params traning_df: dataframe with ImageId, Target columns
    :params image_dir: directory where images are stores, string
    :return: X, y (training array with features and labels)
    """
    # create empty list to store image vectors
    images = []
    # crete empty list to store labels
    targets = []
    # loop over the dataframe
    for index, row in tqdm(training_df.iterrows(),
                            total = len(training_df),
                            desc = "processing images"):
        # get image id
        image_id = row["ImageId"]
        # get image path
        image_path = os.path.join(image_dir, image_id)
        # open the image
        image = Image.open(image_path + ".png")
        # convert the image to numpy array
        image = np.array(image)
        # ravel
        image = image.ravel()
        # append images and target lists
        images.append(image)
        targets.append(int(row["Target"]))
    # convert lists to numpy arrays
    images = np.array(images)
    # print size of this array
    print("images shape: ", images.shape)
    return images, targets

if __name__ == "__main__":
    train_df = pd.read_csv("../input/train.csv")
    train_dir = "../input/train"
    X, y = create_dataset(train_df, train_dir)

    # we create a new column called kfold and fill it with -1
    train_df["kfold"] = -1

    # the next step is to randomize the rows of the data
    train_df = train_df.sample(frac = 1).reset_index(drop = True)

    # fetch labels
    y = train_df.target.values

    # initiate the kfold class from model_selection
    kf = model_selection.StratifiedKFold(n_splits = 5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X= df, y = y)):
        