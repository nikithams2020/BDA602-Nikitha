import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def main():
    # Loading the Dataset

    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"],
    )
    print(df.head(5))

    X_transform = df[
        ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    ].values
    # Y_transform = df["Class"].values
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_transform)
    X = standard_scaler.transform(X_transform)
    print(X)


def mean_diff_cont(X, Y_transform):
    # compute bin edges and bin labels
    b_edges = np.linspace(X.min(), Y_transform.max(), 11)
    bins_l = np.arange(10)

    # compute bin weights
    w = np.histogram(X, bins=b_edges)[0] / len(X)

    # compute the mean of the response variable
    _mean = np.mean(Y_transform)

    # compute the mean of the response variable for each bin
    mean_response = np.zeros_like(bins_l, dtype=float)
    for i in bins_l:
        masking = (X >= b_edges[i]) & (X < b_edges[i + 1])
        mean_response[i] = np.mean(Y_transform[masking])

    # compute the unweighted and weighted difference between the mean of the response variable and the mean of each bin
    unweigh_diff = np.mean(np.square(mean_response - _mean))
    weigh_diff = np.mean(w * np.square(mean_response - _mean))

    print(f"unweighted difference with mean: {unweigh_diff}")
    print(f"unweighted difference with mean: {weigh_diff}")

    # return unweigh_diff, weigh_diff


# Random Forest Classifier feature selection
def random_forest(df, X, Y_transform):
    rf = RandomForestClassifier()
    rf.fit(X, Y_transform)
    print(rf.feature_importances_)


if __name__ == "__main__":
    sys.exit(main())
