import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # Loading the Dataset
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"],
    )
    print(df.head())
    # Summary Statistics
    print("Summary Statistics:", df.describe())
    arr = df[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].values
    print(arr)
    print("Mean of the Iris dataset is:", np.mean(arr, axis=0))
    print("Median of the Iris dataset is:", np.median(arr, axis=0))
    print("Standard Deviation of the Iris dataset is:", np.std(arr, axis=0))
    print("25% Quantile of the Iris dataset is:", np.quantile(arr, q=0.25, axis=0))
    print("50% Quantile of the Iris dataset is:", np.quantile(arr, q=0.50, axis=0))
    print("75% Quantile of the Iris dataset is:", np.quantile(arr, q=0.75, axis=0))
    # Data Visualization
    Iris_fig = px.scatter(
        df, x="Sepal Width", y="Sepal Length", color="Class", symbol="Class"
    )
    Iris_fig.show()

    Iris_fig2 = px.histogram(df, x="Petal Width", nbins=20, color="Class")
    Iris_fig2.update_layout(bargap=0.25)
    Iris_fig2.show()

    Iris_fig3 = px.violin(
        df, y="Sepal Length", color="Class", violinmode="overlay", hover_data=df.columns
    )
    Iris_fig3.show()

    Iris_fig4 = px.bar(df, y="Class", color="Petal Length")
    Iris_fig4.show()

    Iris_fig5 = px.box(
        df, x="Class", y="Petal Length", points="all", notched=True, color="Class"
    )
    Iris_fig5.show()

    # Standard Transformations
    X_transform = df[
        ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    ].values
    Y_transform = df["Class"].values

    standard_scaler = StandardScaler()
    standard_scaler.fit(X_transform)
    X = standard_scaler.transform(X_transform)
    print(X)

    # Model Building
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, Y_transform)
    prediction = random_forest.predict(X)
    probability = random_forest.predict_proba(X)
    print(prediction, probability)

    Logistic_model = LogisticRegression()
    Logistic_model.fit(X, Y_transform)
    prediction = Logistic_model.predict(X)
    probability = Logistic_model.predict_proba(X)
    print(prediction, probability)

    Svc_classifier = SVC()
    Svc_classifier.fit(X, Y_transform)
    prediction = Svc_classifier.predict(X)
    print(prediction)

    pipeline = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    pipeline2 = Pipeline([("StandardScaler", StandardScaler()), ("SVC", SVC())])
    pipeline3 = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("LogisticRegression", LogisticRegression()),
        ]
    )

    pipeline.fit(X_transform, Y_transform)
    pipeline2.fit(X_transform, Y_transform)
    pipeline3.fit(X_transform, Y_transform)
    print("Pipeline 1 score", pipeline.score(X, Y_transform))
    print("Pipeline 2 score", pipeline2.score(X, Y_transform))
    print("Pipeline 3 score", pipeline3.score(X, Y_transform))


if __name__ == "__main__":
    sys.exit(main())
