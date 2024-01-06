import streamlit as st
import pandas as pd
import numpy as np
import base64
import re
import plotly.graph_objects as go
import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes


# Functions ............................................................................................................

collect_numbers = lambda x: [float(i) for i in re.split(',+', x) if i != ""]
collect_numbers_int = lambda x: [int(i) for i in re.split(',+', x) if i != ""]

def filedownload(df):

    """
    filedownload function converts the dataframe df into csv file and downloads it.

    :param df: dataframe containing max_feature, n_estimators, R^2.
    """

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href


def build_model_Adaboost_Regressor(df):

    """
    It builds a model using Adaboost regresion Algorithm.
    Takes input from streamlit web interface and use those inputs for building the model.
    Used GridSearchCV for Hyperparameter Tunning.
    Ploting the result using Plotly Framework.

    :param df: dataframe containing features and labels.
    """

    from sklearn.ensemble import AdaBoostRegressor

    all=False
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    adaboost = AdaBoostRegressor(loss= loss, random_state= random_state)

    grid = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, n_jobs=n_jobs)
    grid.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" %r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" %mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        all = True
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" %rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mae)

    st.write("The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # Grid Data .......
    grid_results = pd.concat(
        [pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],
        axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['learning_rate', 'n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['learning_rate', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot(index='learning_rate', columns='n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Learning_rate')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='Learning_Rate',
                          zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    if all == True:
        criteria = ['RMSE', 'MSE', 'MAE']
        # colors = {'RMSE': 'red',
        #           'MSE': 'orange',
        #           'MAE': 'lightgreen'}
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])

    st.plotly_chart(fig)


    # Change the bar mode
    fig.update_layout(barmode='group')

    # -----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x, y, z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)

##################################################### Linear regression to be worked on
def build_model_Linear_Regressor(df):

    """
    It builds a model using Linear regresion Algorithm.
    Takes input from streamlit web interface and use those inputs for building the model.
    Used GridSearchCV for Hyperparameter Tunning.
    Ploting the result using Plotly Framework.

    :param df: dataframe containing features and labels.
    """

    from sklearn.linear_model import LinearRegression

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    model = LinearRegression()

    if len(ind_var) == 1:
        dfx = X_train[ind_var[0]].values.reshape(-1, 1)
        dfxtest = X_test[ind_var[0]].values.reshape(-1, 1)
        model.fit(dfx, Y_train)
        Y_pred_test = model.predict(dfxtest)

        fig = px.scatter(df, x=ind_var[0], y=Y_test.name, opacity=0.65)
        fig.add_traces(go.Scatter(x=X_test[ind_var[0]], y=Y_pred_test, name='Regression Fit'))
        st.plotly_chart(fig)

    if len(ind_var) == 2:
        dfx = X_train[ind_var]
        model.fit(dfx, Y_train)
        dfxtest = X_test[ind_var]

        mesh_size = .02
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max=X_test[ind_var[0]].min() - margin, X_test[ind_var[0]].max() + margin
        y_min, y_max=X_test[ind_var[1]].min() - margin, X_test[ind_var[1]].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        Y_pred_test = model.predict(dfxtest)

        fig = px.scatter_3d(df, x=ind_var[0], y=ind_var[1], z=Y_test.name)
        fig.update_traces(marker=dict(size=5))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        st.plotly_chart(fig)

    if len(ind_var) > 2:
        dfx = X_train[ind_var]
        model.fit(dfx, Y_train)
        dfxtest = X_test[ind_var]
        Y_pred_test = model.predict(dfxtest)

        st.subheader(f"Visualization shows how {Y_test.name} is dependent on individual variable")
        c = len(ind_var)
        for i in range(0,c):
            dfx = X_train[ind_var[i]].values.reshape(-1, 1)
            dfxtest = X_test[ind_var[i]].values.reshape(-1, 1)
            model.fit(dfx, Y_train)
            pred = model.predict(dfxtest)

            fig = px.scatter(df, x=ind_var[i], y=Y_test.name, opacity=0.65)
            fig.add_traces(go.Scatter(x=X_test[ind_var[i]], y=pred, name='Regression Fit'))
            st.plotly_chart(fig)


    st.subheader('Model Performance')

    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" %r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" %mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" %rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mae)

        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])
        st.plotly_chart(fig)


    # Change the bar mode
    fig.update_layout(barmode='group')

##################################################Randomm Forest
def build_model_RandomForestRegressor(df):

    """
    It builds a model using Adaboost regresion Algorithm.
    Takes input from streamlit web interface and use those inputs for building the model.
    Used GridSearchCV for Hyperparameter Tunning.
    Ploting the result using Plotly Framework.

    :param df: dataframe containing features and labels.
    """

    from sklearn.ensemble import RandomForestRegressor

    all=False
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    # X_train.shape, Y_train.shape
    # X_test.shape, Y_test.shape

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               random_state=random_state,
                               max_features=max_features,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               oob_score=oob_score,
                               n_jobs=n_jobs)

    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)


    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" %r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" %mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        all = True
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mse)
        st.write('Root Mean Squared Error (RMSE):')
        rmse = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" %rmse)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mae)

    st.write("The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # Grid Data .......
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot(index='max_features', columns='n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning (Surface Plot)',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='max_features',
                          zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    st.plotly_chart(fig)

    if all == True:
        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rmse, mse, mae])])

    st.plotly_chart(fig)

    # -----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x, y, z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)

################################################## SVR
def build_model_SVR(df):
    """
        It builds a model using Support Vector regresion Algorithm.
        Takes input from streamlit web interface and use those inputs for building the model.
        Used GridSearchCV for Hyperparameter Tunning.
        Ploting the result using Plotly Framework.

        :param df: dataframe containing features and labels.
        """

    from sklearn.svm import SVR

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    model = SVR()
    if len(ind_var) == 1:
        dfx = X_train[ind_var[0]].values.reshape(-1, 1)
        dfxtest = X_test[ind_var[0]].values.reshape(-1, 1)
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx,Y_train)
        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter(df, x=ind_var[0], y=Y_test.name, opacity=0.65)
        fig.add_traces(go.Scatter(x=X_test[ind_var[0]], y=Y_pred_test, name='Regression Fit'))
        st.plotly_chart(fig)

    if len(ind_var) == 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)

        mesh_size = .02
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max = X_test[ind_var[0]].min() - margin, X_test[ind_var[0]].max() + margin
        y_min, y_max = X_test[ind_var[1]].min() - margin, X_test[ind_var[1]].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter_3d(df, x=ind_var[0], y=ind_var[1], z=Y_test.name)
        fig.update_traces(marker=dict(size=3))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        st.plotly_chart(fig)

    if len(ind_var) > 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        st.subheader(f"Visualization shows how {Y_test.name} is dependent on individual variable")
        c = len(ind_var)
        clf1 = GridSearchCV(model, param_grid)
        for i in range(0,c):
            dfx = X_train[ind_var[i]].values.reshape(-1, 1)
            dfxtest = X_test[ind_var[i]].values.reshape(-1, 1)
            clf1.fit(dfx, Y_train)
            pred = clf1.predict(dfxtest)

            fig = px.scatter(df, x=ind_var[i], y=Y_test.name, opacity=0.65)
            fig.add_traces(go.Scatter(x=X_test[ind_var[i]], y=pred, name='Regression Fit'))
            st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f"
             % (clf.best_params_, clf.best_score_))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" %r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" %mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" %mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" %rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" %mae)

        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])
        st.plotly_chart(fig)

    # st.subheader("Hyperparameter Tuning Results")
    # df_gridsearch = pd.DataFrame(clf.cv_results_)
    # dfViz = df_gridsearch[['param_C', 'param_gamma', 'mean_test_score']]
    #
    # pivot = pd.pivot_table(data=dfViz, index=['param_C'], columns=['param_gamma'], values=['mean_test_score'])
    # sns.heatmap(pivot, annot=True)
    # st.pyplot(plt)


    # Change the bar mode
    fig.update_layout(barmode='group')

################################################## SGD
def build_model_SGD(df):
    """
            It builds a model using Stocastic gradient descent regresion Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
            """

    from sklearn.linear_model import SGDRegressor

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    if scale == 'True':
        from sklearn.preprocessing import StandardScaler

        cols = X_train.columns
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)

    model = SGDRegressor()
    if len(ind_var) == 1:
        dfx = X_train[ind_var[0]].values.reshape(-1, 1)
        dfxtest = X_test[ind_var[0]].values.reshape(-1, 1)
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter(df, x=ind_var[0], y=Y_test.name, opacity=0.65)
        fig.add_traces(go.Scatter(x=X_test[ind_var[0]], y=Y_pred_test, name='Regression Fit'))
        st.plotly_chart(fig)

    if len(ind_var) == 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)

        mesh_size = .02
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max=X_test[ind_var[0]].min() - margin, X_test[ind_var[0]].max() + margin
        y_min, y_max=X_test[ind_var[1]].min() - margin, X_test[ind_var[1]].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter_3d(df, x=ind_var[0], y=ind_var[1], z=Y_test.name)
        fig.update_traces(marker=dict(size=3))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        st.plotly_chart(fig)

    if len(ind_var) > 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        st.subheader(f"Visualization shows how {Y_test.name} is dependent on individual variable")
        c = len(ind_var)
        clf1 = GridSearchCV(model, param_grid)
        for i in range(0, c):
            dfx = X_train[ind_var[i]].values.reshape(-1, 1)
            dfxtest = X_test[ind_var[i]].values.reshape(-1, 1)
            clf1.fit(dfx, Y_train)
            pred = clf1.predict(dfxtest)

            fig = px.scatter(df, x=ind_var[i], y=Y_test.name, opacity=0.65)
            fig.add_traces(go.Scatter(x=X_test[ind_var[i]], y=pred, name='Regression Fit'))
            st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f"
             % (clf.best_params_, clf.best_score_))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')

    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" % r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" % mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" % rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mae)

        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])
        st.plotly_chart(fig)


    # Change the bar mode
    fig.update_layout(barmode='group')

################################################### Kernel Ridge
def build_model_KernelRidge(df):
    """
                    It builds a model using Kernel Ridge Regresion Algorithm.
                    Takes input from streamlit web interface and use those inputs for building the model.
                    Used GridSearchCV for Hyperparameter Tunning.
                    Ploting the result using Plotly Framework.

                    :param df: dataframe containing features and labels.
                    """

    from sklearn.kernel_ridge import KernelRidge

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    if scale == 'True':
        from sklearn.preprocessing import StandardScaler

        cols = X_train.columns
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)

    model = KernelRidge()
    if len(ind_var) == 1:
        dfx = X_train[ind_var[0]].values.reshape(-1, 1)
        dfxtest = X_test[ind_var[0]].values.reshape(-1, 1)
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter(df, x=ind_var[0], y=Y_test.name, opacity=0.65)
        fig.add_traces(go.Scatter(x=X_test[ind_var[0]], y=Y_pred_test, name='Regression Fit'))
        st.plotly_chart(fig)

    if len(ind_var) == 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)

        mesh_size = .02
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max=X_test[ind_var[0]].min() - margin, X_test[ind_var[0]].max() + margin
        y_min, y_max=X_test[ind_var[1]].min() - margin, X_test[ind_var[1]].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter_3d(df, x=ind_var[0], y=ind_var[1], z=Y_test.name)
        fig.update_traces(marker=dict(size=3))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        st.plotly_chart(fig)

    if len(ind_var) > 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        st.subheader(f"Visualization shows how {Y_test.name} is dependent on individual variable")
        c = len(ind_var)
        clf1 = GridSearchCV(model, param_grid)
        for i in range(0, c):
            dfx = X_train[ind_var[i]].values.reshape(-1, 1)
            dfxtest = X_test[ind_var[i]].values.reshape(-1, 1)
            clf1.fit(dfx, Y_train)
            pred = clf1.predict(dfxtest)

            fig = px.scatter(df, x=ind_var[i], y=Y_test.name, opacity=0.65)
            fig.add_traces(go.Scatter(x=X_test[ind_var[i]], y=pred, name='Regression Fit'))
            st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f"
             % (clf.best_params_, clf.best_score_))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')

    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" % r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" % mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" % rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mae)

        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])
        st.plotly_chart(fig)

    # Change the bar mode
    fig.update_layout(barmode='group')

################################################ Elastic Net
def build_model_ElasticNet(df):
    """
                        It builds a model using Elastic Net Regresion Algorithm.
                        Takes input from streamlit web interface and use those inputs for building the model.
                        Used GridSearchCV for Hyperparameter Tunning.
                        Ploting the result using Plotly Framework.

                        :param df: dataframe containing features and labels.
                        """

    from sklearn.linear_model import ElasticNet

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    if scale == 'True':
        from sklearn.preprocessing import StandardScaler

        cols = X_train.columns
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)

    model = ElasticNet()
    if len(ind_var) == 1:
        dfx = X_train[ind_var[0]].values.reshape(-1, 1)
        dfxtest = X_test[ind_var[0]].values.reshape(-1, 1)
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter(df, x=ind_var[0], y=Y_test.name, opacity=0.65)
        fig.add_traces(go.Scatter(x=X_test[ind_var[0]], y=Y_pred_test, name='Regression Fit'))
        st.plotly_chart(fig)

    if len(ind_var) == 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)

        mesh_size = .02
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max=X_test[ind_var[0]].min() - margin, X_test[ind_var[0]].max() + margin
        y_min, y_max=X_test[ind_var[1]].min() - margin, X_test[ind_var[1]].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter_3d(df, x=ind_var[0], y=ind_var[1], z=Y_test.name)
        fig.update_traces(marker=dict(size=3))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        st.plotly_chart(fig)

    if len(ind_var) > 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        st.subheader(f"Visualization shows how {Y_test.name} is dependent on individual variable")
        c = len(ind_var)
        clf1 = GridSearchCV(model, param_grid)
        for i in range(0, c):
            dfx = X_train[ind_var[i]].values.reshape(-1, 1)
            dfxtest = X_test[ind_var[i]].values.reshape(-1, 1)
            clf1.fit(dfx, Y_train)
            pred = clf1.predict(dfxtest)

            fig = px.scatter(df, x=ind_var[i], y=Y_test.name, opacity=0.65)
            fig.add_traces(go.Scatter(x=X_test[ind_var[i]], y=pred, name='Regression Fit'))
            st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f"
             % (clf.best_params_, clf.best_score_))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')

    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" % r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" % mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" % rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mae)

        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])
        st.plotly_chart(fig)


    # Change the bar mode
    fig.update_layout(barmode='group')

################################################# Gradient boosting
def build_model_GradientBoosting(df):
    """
                        It builds a model using Gradient Boosting Regression Algorithm.
                        Takes input from streamlit web interface and use those inputs for building the model.
                        Used GridSearchCV for Hyperparameter Tunning.
                        Ploting the result using Plotly Framework.

                        :param df: dataframe containing features and labels.
                        """

    from sklearn.ensemble import GradientBoostingRegressor


    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    if scale == 'True':
        from sklearn.preprocessing import StandardScaler

        cols = X_train.columns
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)

    model = GradientBoostingRegressor()
    if len(ind_var) == 1:
        dfx = X_train[ind_var[0]].values.reshape(-1, 1)
        dfxtest = X_test[ind_var[0]].values.reshape(-1, 1)
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter(df, x=ind_var[0], y=Y_test.name, opacity=0.65)
        fig.add_traces(go.Scatter(x=X_test[ind_var[0]], y=Y_pred_test, name='Regression Fit'))
        st.plotly_chart(fig)

    if len(ind_var) == 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)

        mesh_size = .02
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max=X_test[ind_var[0]].min() - margin, X_test[ind_var[0]].max() + margin
        y_min, y_max=X_test[ind_var[1]].min() - margin, X_test[ind_var[1]].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        Y_pred_test = clf.predict(dfxtest)

        fig = px.scatter_3d(df, x=ind_var[0], y=ind_var[1], z=Y_test.name)
        fig.update_traces(marker=dict(size=3))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        st.plotly_chart(fig)

    if len(ind_var) > 2:
        dfx = X_train[ind_var]
        dfxtest = X_test[ind_var]
        clf = GridSearchCV(model, param_grid)
        clf.fit(dfx, Y_train)
        Y_pred_test = clf.predict(dfxtest)

        st.subheader(f"Visualization shows how {Y_test.name} is dependent on individual variable")
        c = len(ind_var)
        clf1 = GridSearchCV(model, param_grid)
        for i in range(0, c):
            dfx = X_train[ind_var[i]].values.reshape(-1, 1)
            dfxtest = X_test[ind_var[i]].values.reshape(-1, 1)
            clf1.fit(dfx, Y_train)
            pred = clf1.predict(dfxtest)

            fig = px.scatter(df, x=ind_var[i], y=Y_test.name, opacity=0.65)
            fig.add_traces(go.Scatter(x=X_test[ind_var[i]], y=pred, name='Regression Fit'))
            st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f"
             % (clf.best_params_, clf.best_score_))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')

    st.write('Coefficient of determination ($R^2$):')
    st.info("%0.3f" % r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info("%0.2f" % mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info("%0.2f" % mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info("%0.2f" % rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info("%0.2f" % mae)

        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])])
        st.plotly_chart(fig)

    # Change the bar mode
    fig.update_layout(barmode='group')


# Page Layout ( Streamlit web Interface )
st.set_page_config(page_title="Regression Model Builder")

st.write("""
# Regression Model Builder
""")

# Sidebar ..............................................

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.header("Parameter Configuration")
split_size = st.sidebar.slider('Data Split Ratio (training set)', 10,90,80,5)

st.sidebar.header("Select Regressor")
reg = st.sidebar.selectbox("Choose Regression Algorithm", options=['Linear Regression', 'SVR',
                            'Random Forest Regression', 'Adaboost', 'SGD Regression', 'Kernel Ridge Regression',
                            'Gradient Boosting Regression'])

if reg == 'Random Forest Regression':
    st.sidebar.subheader('Learning Parameters')
    n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    n_estimators_step = st.sidebar.number_input('Step size for n_estimators (n_estimators_step)', 10)
    st.sidebar.write('---')
    max_features = st.sidebar.slider('Max features', 1, 50, (1, 3), 1)
    max_features_step = st.sidebar.number_input('Step Size for max Features', 1)
    st.sidebar.write('---')
    min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)',
                                         1, 10, 2, 1)

    st.sidebar.subheader('General Parameters')
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])
    bootstrap = st.sidebar.selectbox('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    oob_score = st.sidebar.selectbox('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)',
                                     options=[False, True])
    n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(n_estimators[0], n_estimators[1] + n_estimators_step, n_estimators_step)
    max_features_range = np.arange(max_features[0], max_features[1] + max_features_step, max_features_step)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

if reg == 'Adaboost':
    st.sidebar.subheader('Learning Parameters')
    n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    n_estimators_step = st.sidebar.number_input('Step size for n_estimators (n_estimators_step)', 10)
    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])
    lr = [0.0001, 0.001, 0.01, 0.1]
    learning_rate = st.sidebar.select_slider('Range of Learning Rate (learning_rate)',
                                             options=[0.0001, 0.001, 0.01, 0.1], value=(0.0001, 0.01))
    l = lr.index(learning_rate[0])
    r = lr.index(learning_rate[1])
    learning_rate_range = lr[l:r + 1]

    st.sidebar.write('---')

    st.sidebar.header("Loss")
    loss = st.sidebar.selectbox("Choose Loss",options=['linear', 'square', 'exponential'])

    st.sidebar.subheader('General Parameters')
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)

    n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(n_estimators[0], n_estimators[1] + n_estimators_step, n_estimators_step)

    param_grid = dict(learning_rate = learning_rate_range, n_estimators=n_estimators_range)

if reg == 'Linear Regression':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])

if reg == 'SVR':

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])


    st.sidebar.subheader("Hyperparameters for SVR")
    st.sidebar.subheader("Kernel")
    kernel = st.sidebar.selectbox("Enter from the options", options=['All', 'linear', 'rbf', 'poly'])

    numbers = st.sidebar.text_input("Enter values for 'c'. (Separate values with ,)")
    C = collect_numbers(numbers)

    numbers = st.sidebar.text_input("Enter values for 'gamma'. (Separate values with ,)")
    gamma = collect_numbers(numbers)

    numbers = st.sidebar.text_input("Enter values for 'epsilon'. (Separate values with ,)")
    epsilon = collect_numbers(numbers)

    if kernel == 'All':
        kernel = ['linear', 'rbf', 'poly']
    else:
        kernel = [kernel]

    param_grid = dict(kernel = kernel, gamma = gamma, epsilon = epsilon, C = C)

if reg == 'SGD Regression':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])

    st.sidebar.subheader("Standard Scaling")
    scale = st.sidebar.selectbox("Scale the data to be between -1 to 1", options=['True', 'False'])

    st.sidebar.subheader("Hyperparameters for SGD Regressor")
    numbers = st.sidebar.text_input("Enter values for 'alpha'. (Separate values with ,)")
    alpha = collect_numbers(numbers)

    loss = st.sidebar.selectbox("Loss", options=['All', 'squared_loss', 'huber', 'epsilon_insensitive'])
    penalty = st.sidebar.selectbox("Penalty", options=['All', 'l2', 'l1', 'elasticnet'])
    learning_rate = st.sidebar.selectbox("Learning Rate", options=['All', 'constant', 'optimal', 'invscaling'])

    if loss == 'All':
        loss = ['squared_loss', 'huber', 'epsilon_insensitive']
    else:
        loss = [loss]

    if penalty == 'All':
        penalty = ['l2', 'l1', 'elasticnet']
    else:
        penalty = [penalty]

    if learning_rate == 'All':
        learning_rate = ['constant', 'optimal', 'invscaling']
    else:
        learning_rate = [learning_rate]

    param_grid = dict(alpha = alpha, loss = loss, penalty = penalty, learning_rate = learning_rate)

if reg == 'Kernel Ridge Regression':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.subheader("Standard Scaling")
    scale = st.sidebar.selectbox("Scale the data to be between -1 to 1", options=['True', 'False'])

    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Kernel Ridge Regression")
    st.sidebar.subheader("Kernel")
    kernel = st.sidebar.selectbox("Enter from the options", options=['All', 'linear', 'rbf', 'poly'])

    numbers = st.sidebar.text_input("Enter values for 'alpha'. (Separate values with ,)")
    alpha = collect_numbers(numbers)

    numbers = st.sidebar.text_input("Enter values for 'gamma'. (Separate values with ,)")
    gamma = collect_numbers(numbers)

    if kernel == 'All':
        kernel = ['linear', 'rbf', 'poly']
    else:
        kernel = [kernel]

    param_grid = dict(kernel = kernel, gamma = gamma, alpha = alpha)

if reg == 'ElasticNet Regression':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.subheader("Standard Scaling")
    scale = st.sidebar.selectbox("Scale the data to be between -1 to 1", options=['True', 'False'])

    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for ElasticNet Regression")
    st.sidebar.subheader("Selection")
    selection = st.sidebar.selectbox("Enter from the options", options=['All', 'cyclic', 'random'])

    numbers = st.sidebar.text_input("Enter values for 'alpha'. (Separate values with ,)", value='1.0')
    alpha = collect_numbers(numbers)

    numbers = st.sidebar.text_input("Enter values for 'l1_ratio'. (Separate values with ,)", value='0.5')
    l1_ratio = collect_numbers(numbers)

    fit_intercept = st.sidebar.selectbox("Whether the intercept should be estimated or not", options=['Both', 'True', 'False'])
    # if fit_intercept == 'Both' or fit_intercept == 'True':
    #     normalize = st.sidebar.selectbox("Regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm",
    #                                          options=['Both', 'True', 'False'])
    #     if normalize == 'Both':
    #         normalize = ['False', 'True']
    #     else:
    #         normalize = [normalize]

    if selection == 'All':
        selection = ['cyclic', 'random']
    else:
        selection = [selection]

    if fit_intercept == 'Both':
        fit_intercept = ['False', 'True']
    else:
        fit_intercept = [fit_intercept]

    # if fit_intercept.__contains__('True'):
    #     param_grid = dict(selection = selection, l1_ratio = l1_ratio, alpha = alpha,
    #                       fit_intercept = fit_intercept, normalize = normalize)
    # else:
    param_grid = dict(selection=selection, l1_ratio=l1_ratio, alpha=alpha,
                          fit_intercept=fit_intercept)

if reg == 'Gradient Boosting Regression':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.subheader("Standard Scaling")
    scale = st.sidebar.selectbox("Scale the data to be between -1 to 1", options=['True', 'False'])

    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])

    st.sidebar.write('---')
    st.sidebar.header("Hyperparameters for Gradient Boosting Regression")
    st.sidebar.subheader("Loss")
    loss = st.sidebar.selectbox("Enter from the options", options=['All', 'squared_error', 'absolute_error', 'huber',
                                                                   'quantile'])

    st.sidebar.subheader("Learning Rate")
    numbers = st.sidebar.text_input("Enter values for 'learning rate'. (Separate values with ,)", value='0.1')
    learning_rate = collect_numbers(numbers)

    numbers = st.sidebar.text_input("Enter number of estimators. (Separate values with ,)", value='100')
    n_estimators = collect_numbers_int(numbers)

    numbers = st.sidebar.text_input("Enter values for 'Subsample'. (Separate values with ,)", value='1.0')
    subsample = collect_numbers(numbers)

    numbers = st.sidebar.text_input("Enter minimum sample Split. (Separate values with ,)", value='2')
    min_samples_split = collect_numbers_int(numbers)

    numbers = st.sidebar.text_input("Enter minimum samples leaf. (Separate values with ,)", value='1')
    min_samples_leaf = collect_numbers_int(numbers)

    numbers = st.sidebar.text_input("Enter maximum depth. (Separate values with ,)", value='3')
    max_depth = collect_numbers_int(numbers)

    max_features = st.sidebar.selectbox("Maximum Features", options=['All', 'auto', 'sqrt', 'log2'])

    if loss == 'All':
        loss = ['squared_error', 'absolute_error', 'huber', 'quantile']
    else:
        loss = [loss]

    if max_features == 'All':
        max_features = ['auto', 'sqrt', 'log2']
    else:
        max_features = [max_features]

    param_grid = dict(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                      max_depth=max_depth, max_features=max_features)


# main Body ...............................................................................................

st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    if reg == 'Random Forest Regression':
        build_model_RandomForestRegressor(df)
    if reg == 'Adaboost':
        build_model_Adaboost_Regressor(df)
    if reg == 'Linear Regression':
        build_model_Linear_Regressor(df)
    if reg == 'SVR':
        build_model_SVR(df)
    if reg == 'SGD Regression':
        build_model_SGD(df)
    if reg == 'Kernel Ridge Regression':
        build_model_KernelRidge(df)
    # if reg == 'ElasticNet Regression':
    #     build_model_ElasticNet(df)
    if reg == 'Gradient Boosting Regression':
        build_model_GradientBoosting(df)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df.head(5))

        if reg == 'Random Forest Regression':
            build_model_RandomForestRegressor(df)
        if reg == 'Adaboost':
            build_model_Adaboost_Regressor(df)
        if reg == 'Linear Regression':
            build_model_Linear_Regressor(df)
        if reg == 'SVR':
            build_model_SVR(df)
        if reg == 'SGD Regression':
            build_model_SGD(df)
        if reg == 'Kernel Ridge Regression':
            build_model_KernelRidge(df)
        # if reg == 'ElasticNet Regression':
        #     build_model_ElasticNet(df)
        if reg == 'Gradient Boosting Regression':
            build_model_GradientBoosting(df)