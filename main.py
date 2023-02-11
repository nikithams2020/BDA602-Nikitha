import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# https://scikit-learn.org/stable/getting_started.html
# https://www.youtube.com/watchcmqM&t=348s
# https://plotly.com/python/multiple-axes
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
        df,
        x="Sepal Width",
        y="Sepal Length",
        color="Class",
        symbol="Class",
        title="Sepal Width vs Sepal Length wrt to Species",
    )
    Iris_fig.write_html(file="viz_plots.html", include_plotlyjs="cdn")

    Iris_fig2 = px.histogram(
        df,
        x="Petal Width",
        nbins=20,
        color="Class",
        title="Petal Width vs Class/Species",
    )
    Iris_fig2.update_layout(bargap=0.25)
    Iris_fig2.write_html(file="viz_plots.html", include_plotlyjs="cdn")

    Iris_fig3 = px.violin(
        df,
        y="Sepal Length",
        color="Class",
        violinmode="overlay",
        hover_data=df.columns,
        title="Sepal Length vs Class/Species",
    )
    Iris_fig3.write_html(file="viz_plots.html", include_plotlyjs="cdn")

    Iris_fig4 = px.bar(
        df, y="Class", color="Petal Length", title="Petal length vs Class/Species"
    )
    Iris_fig4.write_html(file="viz_plots.html", include_plotlyjs="cdn")

    Iris_fig5 = px.box(
        df,
        x="Class",
        y="Petal Length",
        points="all",
        notched=True,
        color="Class",
        title="Petal Length vs Class/Species",
    )
    Iris_fig5.write_html(file="viz_plots.html", include_plotlyjs="cdn")

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


# Response Plots
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"],
)
types = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
col = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

for type in types:
    for i in col:
        iris_response = df[[i, "Class"]]

        a = np.array(iris_response[i])
        population, bins = np.histogram(a, bins=10, range=(np.min(a), np.max(a)))
        bins_mod = 0.5 * (bins[:-1] + bins[1:])

        sl_iris = iris_response.loc[iris_response["Class"] == type]
        b = np.array(sl_iris[i])
        population_Iris, _ = np.histogram(b, bins=bins)

        p_response = population_Iris / population

        response_rate = len(df.loc[df["Class"] == type]) / len(df)
        response_rate_arr = np.array([response_rate] * len(bins_mod))

        print(response_rate_arr)

        fig = go.Figure(
            data=go.Bar(
                x=bins_mod,
                y=population,
                name=i,
                marker=dict(color=" skyblue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=bins_mod,
                y=p_response,
                yaxis="y2",
                name="Response",
                marker=dict(color="red"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=bins_mod,
                y=response_rate_arr,
                yaxis="y2",
                mode="lines",
                name=f"Species {type}",
            )
        )

        fig.update_layout(
            title_text=f"{type} vs {i} Mean of Response Rate Plot",
            legend=dict(orientation="v"),
            yaxis=dict(
                title=dict(text="Total Population"),
                side="left",
                range=[0, 30],
            ),
            yaxis2=dict(
                title=dict(text="Response"),
                side="right",
                range=[-0.1, 1.2],
                overlaying="y",
                tickmode="auto",
            ),
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Predictor Bins")

        fig.write_html(file="viz_plots.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    sys.exit(main())
