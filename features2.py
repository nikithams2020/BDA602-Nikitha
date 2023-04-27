import os
import re
import logging
import warnings
import sqlalchemy
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import plotly.express as px
import statsmodels.api as sm
# import cat_correlation as co_rel
import plotly.graph_objects as go
import plotly.figure_factory as ff
# from dataset_loader import TestDatasets
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

global cont_url_dict
global cat_url_dict


def getfilename(a):
    path = Path(a)
    d = path.parent.absolute()
    z = a.replace(os.fspath(d), '')
    filename = re.search('[a-z]+', z).group()
    return filename


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, getfilename(val))


class response_table():

    def __init__(self, df):
        self.df = df

    def get_feature_importance(self, pred_var, response):
        X = sm.add_constant(self.df[pred_var])  # Added a constant term to the X variable
        model = sm.OLS(self.df[response], X).fit()  # Fit the model
        t_score = model.tvalues[1]  # Get the t score for X
        p_value = model.pvalues[1]  # Get the p-value for X
        return t_score, p_value

    def random_variable(self, cont_var, response):
        rf = RandomForestClassifier()
        df1 = self.df[cont_var]
        rf.fit(df1, self.df[response])
        print(rf.feature_importances_)

        feat_imp = rf.feature_importances_
        feat_names = df1.columns
        feat_rank = pd.DataFrame({"columns": feat_names, "impact": feat_imp})
        feat_rank = feat_rank.sort_values("impact", ascending=False).reset_index(drop=True)
        # Print rankings table
        return feat_rank


def plotting_cont(df, pred_var, response):
    table = pd.DataFrame(columns=["predictor", "file_link"])
    data_response = df[[pred_var, response]]
    bin_1 = len(data_response[pred_var].unique()) if len(data_response[pred_var].unique()) < 10 else 10
    a = np.array(data_response[pred_var])
    population_a, bins = np.histogram(a, bins=bin_1, range=(np.min(a), np.max(a)))
    bins_mod = 0.5 * (bins[:-1] + bins[1:])

    res = data_response.loc[data_response[response] == 1]
    b = np.array(res[pred_var])
    population_b, _ = np.histogram(b, bins=bins)

    p_response = population_b / population_a

    response_rate = len(df.loc[df[response] == 1]) / len(df)
    response_rate_arr = np.array([response_rate] * len(bins_mod))

    print(response_rate_arr)

    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=population_a,
            name=pred_var,
            marker=dict(color="skyblue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=p_response,
            yaxis="y2",
            name="mui-mupop",
            marker=dict(color="red")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=response_rate_arr,
            yaxis="y2",
            mode="lines",
            name=f"population mean",
        )
    )

    fig.update_layout(
        title_text=f"{pred_var} vs {response} Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Total Population"),
            side="right",
            range=[0, max(population_a) + 1]),
        # xaxis=dict(
        #     range=[0, max(bins_mod)+5]
        # ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="left",
            range=[min(p_response) - 0.05, max(p_response) + 0.1],
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")
    filename = f"{pred_var}_cont_res.html"
    fig.write_html(file=filename, include_plotlyjs="cdn")
    table.loc[len(table)] = [pred_var, filename]
    return table


def cat_cont_res(df, pred_var, response):
    table = pd.DataFrame(columns=["predictor", "file_link"])
    data_response = df[[pred_var, response]][df[response] == 1]
    population_a, bins = df[pred_var].value_counts(), [i for i in range(0, len(df[pred_var].unique()))]

    res = data_response.loc[data_response[response] == 1]
    b = np.array(res[response])
    population_b = data_response[pred_var].value_counts()

    p_response = population_b / population_a

    response_rate = len(df.loc[df[response] == 1]) / len(df)
    response_rate_arr = np.array([response_rate])

    print(response_rate_arr)

    fig = go.Figure(
        data=go.Bar(
            x=df[pred_var].unique(),
            y=population_a,
            name=pred_var,
            marker=dict(color="skyblue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[pred_var].unique(),
            y=p_response,
            yaxis="y2",
            name="mui-mupop",
            marker=dict(color="red")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[pred_var].unique(),
            y=[response_rate_arr[0] for _ in range(0, len(df[pred_var].unique()))],
            yaxis="y2",
            mode="lines",
            name=f"population mean",
        )
    )

    fig.update_layout(
        title_text=f"{pred_var} vs {response} Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Total Population"),
            side="right",
            range=[0, max(population_a) + 1]),
        # xaxis=dict(
        #     range=[0, max(bins_mod)+5]
        # ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="left",
            range=[min(p_response) - 0.05, max(p_response) + 0.1],
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")
    filename = f"{pred_var}_cat_res.html"
    fig.write_html(file=filename, include_plotlyjs="cdn")
    table.loc[len(table)] = [pred_var, filename]
    return table


def DisplayTableTemplate(df, pred_var, response, is_cont=False):
    table = pd.DataFrame(columns=["predictor", "file_link"])
    if is_cont:
        fig = px.violin(df, x=response, y=pred_var, box=True, color=response)
        fig.update_layout(
            title=f"Variable: {pred_var}",
            yaxis_title=f"{response} ({pred_var})",
            xaxis_title=f"Groupings (survived)",
        )
        filename = f"{pred_var}_cont_base.html"
    else:
        df1 = df.groupby([pred_var, response]).count().iloc[:, 1].reset_index()
        fig = go.Figure(data=go.Heatmap(y=df1[pred_var], x=df1[response], z=df1.iloc[:, -1],
                                        text=df1.iloc[:, -1], colorscale='Viridis', hoverongaps=False,
                                        texttemplate="%{text} (pop: %{text}) ",
                                        textfont={"size": 16}))
        fig.update_layout(
            title=f"Variable: {pred_var}",
            yaxis_title=f"pred_var",
            xaxis_title=f"Groupings (survived)",
        )
        filename = f"{pred_var}_cat_base.html"
    fig.write_html(file=filename, include_plotlyjs="cdn")

    table.loc[len(table)] = [pred_var, filename]
    return table


def continuous_continuous_pairs(df, cont_var):
    global cont_url_dict
    pearson_df = pd.DataFrame(columns=['cont_1', 'cont_2', 'corr'])
    s = sns.heatmap(
        df[cont_var].corr(),
        annot=True,
        center=0,
        cmap="Blues",
    )
    s = s.get_figure()
    s.savefig("Continuous_VS_Continuous_correlation_matrix.png")

    col_tracker = set()

    k = 0
    for i in cont_var:
        for j in cont_var:
            if i != j:
                print(i, j, k)
                k += 1
                corr = df[[i, j]].corr(method='pearson')[i][j]
                pearson_df.loc[k] = [i, j, corr]

    url_df = pd.DataFrame(cont_url_dict).T.reset_index()
    url_df.columns = ['predictor', 'file_links']
    temp_df = pearson_df.merge(url_df, left_on='cont_1', right_on='predictor', how='left')
    url_df.rename(columns={'file_links': 'links'}, inplace=True)
    temp_df = temp_df.merge(url_df, left_on='cont_2', right_on='predictor', how='left')
    temp_df.drop(['predictor_y', 'predictor_x'], axis=1, inplace=True)
    return temp_df


def temp(df, response, cont_var):
    # Correlation table logic
    table = pd.DataFrame()
    corr_values = pd.DataFrame(
        columns=["predictor_1", "predictor_2", "corr_value", "abs_corr_value"]
    )
    temp_corr = set()

    for e, i in enumerate(cont_var):

        for j in cont_var:
            if i == j:
                continue
            else:
                corr = df[[i, j]].corr(method='pearson')[i][j]

                if corr not in temp_corr:
                    temp_corr.add(corr)
                    corr_values.loc[e] = [i, j, corr, abs(corr)]

        # t_score, p_value = get_feature_importance(df,i,response)
        temp_table = DisplayTableTemplate(df, i, response, True)
        table = pd.concat([table, temp_table])
    return table, corr_values


def create_table(final_table, file_name, clickable, sort_by):
    plot_click = {i: make_clickable for i in clickable}
    table = final_table.style.set_properties(**{"border": "1.6px dotted purple"}).format(
        plot_click, escape="html")
    # table.set_properties( "border-collapse": "collapse;"})
    table.background_gradient(axis=0, gmap=final_table[sort_by], cmap='YlGnBu')
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: gold; color: black;'}

    table.set_table_styles([{'selector': '*', 'props':
        [('color', 'black'), ('border-style', 'solid'), ('border-width', '1px'), ('border-collapse', 'collapse')]}])
    table.set_table_styles([headers])

    cont_cont_table = table.to_html(file_name, index=False)  # "Continuous-Continuous.html"


def mean_of_response_cont_cont(X, continuous, response):
    # Predictor1 = []
    # Predictor2 = []
    res = []
    output = {"column": [], "un_weigh": [], "weigh": []}
    Unweighted_Mean_of_Response = []
    Weighted_Mean_of_Response = []
    urls = []
    bin_size = 16
    bin_size_2d = bin_size * bin_size
    for i in range(0, len(continuous)):
        # for j in range(0, len(continuous)):
        #     if i != j and i < j:
        population_mean = X[response].mean()
        population_count = X[response].count()
        bins = {}
        bins["cont1_bins"] = pd.cut(X[continuous[i]], bin_size)
        # bins["cont2_bins"] = pd.cut(X[continuous[response]], bin_size)
        bins_df = pd.DataFrame(bins)
        bin_columns = bins_df.columns.to_list()
        print(bins_df)
        filtered_df = X.filter([continuous[i], response], axis=1)
        joined_bin = filtered_df.join(bins_df)
        print(joined_bin)

        grouped_bin = joined_bin.groupby(bin_columns)
        bin_mean = grouped_bin.mean().unstack()
        print(bin_mean)

        counts = grouped_bin.count().unstack()
        response_count = counts[response]
        res_means_df = bin_mean[response]
        res_means_diff_df = res_means_df - population_mean
        res_weights_df = response_count / population_count

        res_means_diff_weighted_df = res_means_diff_df * res_weights_df

        diff_mean_unweighted = (
                                       res_means_diff_df.pow(2).sum() / bin_size_2d
                               ) ** 0.5
        diff_mean_weighted = (
                                     res_means_diff_weighted_df.pow(2).sum() / bin_size_2d
                             ) ** 0.5
        Unweighted_Mean_of_Response.append(diff_mean_unweighted)
        Weighted_Mean_of_Response.append(diff_mean_weighted)
        output['column'].append(continuous[i])
        output['un_weigh'].append(diff_mean_unweighted)
        output['weigh'].append(diff_mean_weighted)

    return output


def mean_res_cat(df, cat_var, response):
    cat_diff_table = pd.DataFrame(
        columns=["predictor", "Diff Mean Response (Weighted)", "Diff Mean Response (Unweighted)"]
    )
    for p in cat_var:
        w = df.groupby(df[p]).count()[response] / sum(
            df.groupby(df[p]).count()[response]
        )
        w = w.reset_index()

        _mean = np.mean(df[response])
        bin = df.groupby(df[p]).apply(np.mean)
        response_mean = bin[response].reset_index(name="Mean: response")
        unweigh_diff = np.sum(np.square(response_mean.iloc[:, 1] - _mean)) / len(
            response_mean
        )
        weigh_diff = (
                np.sum(w.iloc[:, 1] * np.square(response_mean.iloc[:, 1] - _mean))
                / len(response_mean)
                * response_mean["Mean: response"].count()
        )

        cat_diff_table.loc[len(cat_diff_table)] = [p, unweigh_diff, weigh_diff]
    return cat_diff_table


def model_build_eval(df, pred, res):
    df.fillna(0, inplace=True)
    feat = df[pred]
    target = df[res]
    X_train, X_test, Y_train, Y_test = train_test_split(feat, target, test_size=0.25, shuffle=True, random_state=42)

    GBM_CLF = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=0).fit(X_train,
                                                                                                                Y_train)
    GBM_accu = GBM_CLF.score(X_test, Y_test)
    GBM_pred = GBM_CLF.predict(X_test)
    print(classification_report(Y_test, GBM_pred))
    print("Gradient boosting machine's accuracy: ", GBM_accu)

    dT_class = DecisionTreeClassifier()
    dT_class.fit(X_train, Y_train)
    Y_pred = dT_class.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print("Decision Tree's accuracy: ", dT_class.score(X_test, Y_test))


def main(logger):
    db_user = "admin"
    db_pass = "1234"
    db_host = "localhost"
    db_database = "baseball"
    connect_string = f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """ SELECT * FROM baseball_features  """

    df = pd.read_sql(query, con=sql_engine)

    df.fillna(0, inplace=True)
    predictor = list(df.columns)
    response = "HomeTeamWins"
    predictor.remove(response)
    cont_var, cat_var = [], []
    cont_temp_table = pd.DataFrame()
    temp_df = {"col": [], "p_value": [], "t_value": []}

    for i in df.columns:
        if i in predictor:
            if df[i].dtype == "object":
                cat_var.append(i)
                logger.info(f"{i} is a categorical variable")
            else:
                if df[i].dtype == "bool":
                    df[i] = df[i].map({True: 1, False: 0})
                cont_var.append(i)
                logger.info(f"{i} is a continuous variable")

    res_tb = response_table(df)
    random_score = res_tb.random_variable(cont_var, response=response)

    for i in cont_var:
        t_value, p_value = res_tb.get_feature_importance(i, response=response)
        cont_temp_table_1 = plotting_cont(df, i, response=response)
        cont_temp_table = pd.concat([cont_temp_table, cont_temp_table_1])
        temp_df['col'].append(i)
        temp_df['p_value'].append(p_value)
        temp_df['t_value'].append(t_value)

    cont_temp_table["file_link"] = cont_temp_table["file_link"].apply(lambda x: os.path.join(os.getcwd(), x))
    global cont_url_dict
    cont_url_dict = {row['predictor']: [row['file_link']] for i, row in cont_temp_table.iterrows()}
    data = pd.DataFrame(temp_df)
    output = mean_of_response_cont_cont(df, cont_var, response)
    mean_of_res = pd.DataFrame(output)
    first_table = data.merge(random_score, left_on='col', right_on='columns', how='inner').merge(mean_of_res,
                                                                                                 left_on='columns',
                                                                                                 right_on='column',
                                                                                                 how='inner').drop(
        ['columns', 'column'], axis=1)
    first_table = first_table[['col', 'weigh', 'un_weigh', 'impact', 'p_value', 't_value']]
    first_table = first_table.rename(columns={'col': 'Feature', 'weigh': "Diff Mean Response (Weighted)",
                                              'un_weigh': "Diff Mean Response (Unweighted)",
                                              'impact': 'Random Forest Variable Importance', 'p_value': 'P-Value',
                                              't_value': 'T-Score', 'impact': 'Random Forest Variable Importance'})
    model_build_eval(df, predictor, response)
    table, _ = temp(df, response, cont_var)
    table["file_link"] = table["file_link"].apply(lambda x: os.path.join(os.getcwd(), x))
    cont_temp_table = table.merge(cont_temp_table, on='predictor', how='inner').merge(first_table, left_on='predictor',
                                                                                      right_on='Feature', how='inner')
    cont_temp_table.rename(
        columns={
            "file_link_y": "Mean of Response Plot",
            "file_link_x": "Plot"
        }, inplace=True)
    click1 = ['Plot', "Mean of Response Plot"]
    cont_temp_table.drop('Feature', axis=1, inplace=True)
    cont_temp_table.rename(columns={'predictor': 'Feature'}, inplace=True)
    create_table(cont_temp_table, 'cont_plot.html', click1, sort_by='Random Forest Variable Importance')
    if len(cat_var) > 0:
        global cat_url_dict
        cat_url_dict = {}
        cat_temp_table = pd.DataFrame()
        for i in cat_var:
            cat_temp_table_z = pd.DataFrame()
            temp_table = DisplayTableTemplate(df, i, response, is_cont=False)
            temp_table["file_link"] = temp_table["file_link"].apply(lambda x: os.path.join(os.getcwd(), x))
            for x, row in temp_table.iterrows():
                if row['predictor'] not in cat_url_dict:
                    cat_url_dict[row['predictor']] = [row['file_link']]
            cat_temp_table_z = pd.concat([cat_temp_table_z, temp_table], axis=0)
            table = cat_cont_res(df, i, response)
            cat_temp_table_1 = cat_temp_table_z.merge(table, on='predictor', how='inner')
            cat_temp_table = pd.concat([cat_temp_table, cat_temp_table_1], axis=0)
        cat_temp_table.rename(columns={'file_link_x': 'Plot', 'file_link_y': 'Mean of Response Plot'}, inplace=True)
        mean_of_res_cat = mean_res_cat(df, cat_var, response)
        second_table = cat_temp_table.merge(mean_of_res_cat, on='predictor', how='inner')
        click2 = ['Plot', 'Mean of Response Plot']
        create_table(second_table, 'cat_plot.html', click2, sort_by='Diff Mean Response (Weighted)')

    pearson_corr = continuous_continuous_pairs(df, cont_var)
    pearson_corr.rename(columns={'file_links': 'cont_1_url', 'links': 'cont_2_url'}, inplace=True)
    click4 = ['cont_1_url', 'cont_2_url']
    create_table(pearson_corr, 'pearson_corr.html', click4, sort_by='corr')

    del temp_df

    cont_cont_brute_force_table = pd.DataFrame(
        columns=[
            "cont_1",
            "cont_2",
            "diff_mean_resp_ranking",
            "diff_mean_resp_weighted_ranking",
            "pearson",
            "abs_pearson",
            "link"
        ]
    )

    y = response
    cont_cont_brute_force_table_temp = pd.DataFrame()
    # Cont-Cont Brute force
    for c1 in cont_var:
        for c2 in cont_var:
            df_temp = df  # [[p1,p2]]
            if c1 != c2:
                c1_binning, c2_binning = "Bins:" + c1, "Bins:" + c2

                df[c1_binning] = pd.cut(x=df[c1], bins=10)
                df[c2_binning] = pd.cut(x=df[c2], bins=10)

                mean = {y: np.mean}
                length = {y: np.size}
                mean_values = (
                    df.groupby([c1_binning, c2_binning])
                    .agg(mean)
                    .reset_index()
                )
                lengths = (
                    df.groupby([c1_binning, c2_binning])
                    .agg(length)
                    .reset_index()
                )

                mean_values_merged = pd.merge(
                    mean_values,
                    lengths,
                    how="left",
                    left_on=[c1_binning, c2_binning],
                    right_on=[c1_binning, c2_binning],
                )

                mean_values_merged["population_mean"] = df[response].mean()
                mean_values_merged["population"] = mean_values_merged[
                    response + "_y"
                    ].sum()

                mean_values_merged["diff with mean of response"] = (
                        mean_values_merged["population_mean"]
                        - mean_values_merged[response + "_x"]
                )

                mean_values_merged["squared diff"] = (
                        1
                        / 100
                        * np.power(mean_values_merged["diff with mean of response"], 2)
                )

                mean_values_merged["weighted diff with mean of response"] = (
                        np.power((mean_values_merged["diff with mean of response"]), 2)
                        * mean_values_merged[response + "_y"]
                        / mean_values_merged["population"]
                )

                unweigh_diff = [mean_values_merged["squared diff"].sum()]
                weigh_diff = [
                    mean_values_merged["weighted diff with mean of response"].sum()
                ]
                t1 = [round(i, 4) for i in mean_values_merged["diff with mean of response"].tolist() if
                      i not in [np.nan, float('Nan')]]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=mean_values_merged["diff with mean of response"].tolist(),
                        x=mean_values_merged[c1_binning].astype(str).tolist(),
                        y=mean_values_merged[c2_binning].astype(str).tolist(),
                        text=t1,
                        colorscale='Viridis', hoverongaps=False,
                        texttemplate="%{text}", textfont={"size": 16}
                    )
                )
                fig.update_layout(
                    title="Brute Force" + y,
                    xaxis_title=c1 + " bins",
                    yaxis_title=c2 + " bins",
                )

                filename = f"{c1}_{c2}_brute_force.html"

                fig.write_html(file=filename, include_plotlyjs="cdn")

                corr = df[[c1, c2]].corr()[c1][c2]
                cont_cont_brute_force_table.loc[len(cont_cont_brute_force_table)] = [
                    c1,
                    c2,
                    unweigh_diff,
                    weigh_diff,
                    corr,
                    abs(corr),
                    os.getcwd() + "/" + filename
                ]
                cont_cont_brute_force_table_temp = pd.concat(
                    [cont_cont_brute_force_table_temp, cont_cont_brute_force_table], axis=0).reset_index(drop=True)
    create_table(cont_cont_brute_force_table_temp, 'brute_force_cont_cont.html', ['link'], sort_by='pearson')


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info('started')
    main(logging)
    logging.info('Completed!!!...')