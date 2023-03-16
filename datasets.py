import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"],
)
print(df.head())
# df = pd.read_csv("heart.csv")
# df.head()
# def response_checker():
bool_cols = df.select_dtypes(include="bool").columns
cont_cols = df.select_dtypes(include=["float", "int"]).columns
print(bool_cols)
print(cont_cols)

# categorical or continuous checking
for col_name, col_data in df.items():
    if col_data.dtype == "object":
        print(f"{col_name} is a categorical variable")
    elif np.issubdtype(col_data.dtype, np.number):
        print(f"{col_name} is a continuous variable")

# Plotting for continuous and categorical values
fig = px.scatter_matrix(df)
fig.update_layout(
    title="Scatterplot Matrix", xaxis=dict(title="X Axis"), yaxis=dict(title="Y Axis")
)
fig.show()

for column in df.columns:
    if df[column].dtype == "float64" or df[column].dtype == "int64":
        fig = px.box(df, y=column)
        fig.update_layout(title=f"Boxplot of {column}", yaxis_title=column)
        fig.show()

# Linear Regression p value and T score
X = df.loc[:, df.columns != "target"]
Y = df["target"]
X = sm.add_constant(X)  # Add a constant term to the X variable
model = sm.OLS(Y, X).fit()  # Fit the model
t_score = model.tvalues[1]  # Get the t score for X
p_value = model.pvalues[1]  # Get the p-value for X


# df = pd.read_csv("heart.csv")
# df.head()
