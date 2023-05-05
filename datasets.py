import sys

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

from dataset_loader import TestDatasets


def main():
    df, predictors, response = TestDatasets().get_test_data_set(data_set_name="titanic")
    print(predictors)
    print(df[response])

    # df = pd.read_csv(
    # "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    # names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"],
    # )
    print(df.head())

    # checking the col types
    bool_cols = df.select_dtypes(include="bool").columns
    cont_cols = df.select_dtypes(include=["float", "int"]).columns
    print(bool_cols)
    print(cont_cols)
    dt = pd.DataFrame()
    df1 = df.copy()

    a, b = [], []

    # categorical or continuous checking
    for col_name, col_data in df.items():
        if col_data.dtype == "object":
            a.append(col_name)
            temp = pd.get_dummies(df[col_name], drop_first=True)
            print(temp)
            df1.drop(f"{col_name}", axis=1, inplace=True)
            dt = pd.concat([temp, dt], axis=1)
            print(f"{col_name} is a categorical variable")
        elif np.issubdtype(col_data.dtype, np.number):
            b.append(col_name)
            print(f"{col_name} is a continuous variable")
    df1 = pd.concat([dt, df1], axis=1)
    temp = pd.get_dummies(df["adult_male"], drop_first=True)
    df1.drop("adult_male", axis=1, inplace=True)
    df1 = pd.concat([temp, df1], axis=1)
    print("nikitha")
    print(df1)

    # a = df[predictors].dtypes
    # print(a)
    # b = df[response].dtypes
    # print(b)

    # Plotting for continuous and categorical values
    for i in a:
        fig = px.scatter_matrix(df)
        fig.update_layout(
            title="Scatterplot Matrix",
            xaxis=dict(title="X Axis"),
            yaxis=dict(title="Y Axis"),
        )
    fig.show()

    for column in df.columns:
        if df[column].dtype == "float64" or df[column].dtype == "int64":
            fig = px.box(df, y=column)
            fig.update_layout(title=f"Boxplot of {column}", yaxis_title=column)
            fig.show()

    # Linear Regression p value and T score
    # last_column = df.shape[1] - 1
    # X = df.iloc[:, :last_column]
    # Y = df.iloc[:, last_column

    # linear_regression
    # print(predictors, response)
    X = sm.add_constant(df1)
    print(X)  # Add a constant term to the X variable
    model = sm.OLS(df[response], X).fit()  # Fit the model
    t_score = model.tvalues  # Get the t score for X
    p_value = model.pvalues  # Get the p-value for X
    print(f"t-score: {t_score}")
    print(f"p-value: {p_value}")

    # logistic regression
    X = sm.add_constant(df1)
    print(X)  # Add a constant term to the X variable
    model = sm.Logit(df[response], X).fit()  # Fit the model
    t_score = model.tvalues  # Get the t score for X
    p_value = model.pvalues  # Get the p-value for X
    print(f"t-score-logit: {t_score}")
    print(f"p-value-logit: {p_value}")

    # def random_forest(df, X, Y_transform):
    rf = RandomForestClassifier()
    rf.fit(df1, df[response])
    print(rf.feature_importances_)

    feat_imp = rf.feature_importances_
    feat_names = df1.columns
    feat_rank = pd.DataFrame({"impact": feat_imp, "columns": feat_names})
    feat_rank = feat_rank.sort_values("impact", ascending=False)

    # Print rankings table
    print(feat_rank)

    # #def mean_diff_cont(df1, df[response]):
    b_edges = np.linspace(df1.min(), df[response].max(), 11)
    bins_l = np.arange(10)

    # compute bin weights
    w = np.histogram(df[response], bins=bins_l)[0] / len(df1)
    # compute the mean of the response variable
    _mean = np.mean(df[response])

    # compute the mean of the response variable for each bin
    mean_response = np.zeros_like(bins_l, dtype=float)
    for i in bins_l:
        masking = (df[response] >= b_edges[i]) & (df[response] < b_edges[i + 1])
        print(masking)
        mean_response[i] = np.mean(df[response][masking])

    # compute the unweighted and weighted difference between the mean of the response variable and the mean of each bin
    unweigh_diff = np.mean(np.square(mean_response - _mean))
    weigh_diff = np.mean(w * np.square(mean_response - _mean))

    print(f"unweighted difference with mean: {unweigh_diff}")
    print(f"unweighted difference with mean: {weigh_diff}")

    # def mean_cat_diff(df, predictor, response):
    # Calculate overall mean of response variable
    overall_mean = df[response].mean()

    # Calculate group means of response variable
    group_means = df.groupby(df1)[response].mean().reset_index()

    # Calculate unweighted difference between group means and overall mean
    unweighted_diff = np.mean(np.square(group_means[response] - overall_mean))

    # Calculate weighted difference between group means and overall mean
    weights = df.groupby(df1).size() / len(df)
    weighted_diff = np.sum(weights * np.square(group_means[response] - overall_mean))

    # Print results
    print(f"Unweighted difference with mean: {unweighted_diff}")
    print(f"Weighted difference with mean: {weighted_diff}")

    # mean response plots for different categories
    mean_response = df[response].mean()
    df["diff"] = df[response] - mean_response
    if df1 == "categorical":
        fig = px.bar(
            df, df1, df[response], title="mean of response of categorical values"
        )

    else:
        fig = px.scatter(
            df, df1, df[response], title="mean of response of numerical values"
        )


# df = pd.read_csv("heart.csv")
# df.head()

if __name__ == "__main__":
    sys.exit(main())
