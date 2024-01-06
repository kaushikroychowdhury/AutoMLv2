import streamlit as st
import pandas as pd
import numpy as np
import base64
import re

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.figure_factory import create_annotated_heatmap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

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

def DecisionBoundary(model, X, Y):
    mesh_size=.02
    margin=1
    x_min, x_max=X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    # xrange=np.arange(x_min, x_max, mesh_size)
    # yrange=np.arange(y_min, y_max, mesh_size)
    xx, yy=np.meshgrid(np.arange(x_min, x_max, mesh_size)
                       , np.arange(y_min, y_max, mesh_size))
    y_=np.arange(y_min, y_max, mesh_size)
    # Run model
    pred=model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred=pred.reshape(xx.shape)

    fig=make_subplots(rows=1, cols=1)

    trace1=go.Heatmap(x=xx[0], y=y_, z=pred,
                      colorscale='viridis',
                      showscale=False)
    # for test decision Boundary
    # trace2=go.Scatter(x=X_test[:, 0], y=X_test[:, 1],
    #                   mode='markers',
    #                   showlegend=False,
    #                   marker=dict(size=10,
    #                               color=Y_test,
    #                               colorscale='Viridis',
    #                               line=dict(color='black', width=1))
    #                   )

    # for train decision boundary
    trace2=go.Scatter(x=X[:, 0], y=X[:, 1],
                      mode='markers',
                      showlegend=False,

                      marker=dict(size=10,
                                  color=Y,
                                  colorscale='viridis',
                                  line=dict(color='black', width=1))
                      )

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)

    return fig

def ROC_AUC(y_scores, y_onehot):
    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig=go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(y_scores.shape[1]):
        y_true=y_onehot.iloc[:, i]
        y_score=y_scores[:, i]

        fpr, tpr, _=roc_curve(y_true, y_score)
        if check == 'True':
            auc_score=roc_auc_score(y_true, y_score, multi_class=multi_class)
        else:
            auc_score=roc_auc_score(y_true, y_score, multi_class=multi_class)

        name=f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    return fig

def ConfusionMatrix(cm, z_text, d):
    fig=create_annotated_heatmap(cm, x=d, y=d, annotation_text=z_text, colorscale='purp')
    fig.update_layout(title_text='Confusion matrix', xaxis_title='Predicted value',
        yaxis_title='True Value')
    fig['data'][0]['showscale']=True

    return fig

def getIndex(df):
    indices = list()
    for col_name in ind_var:
        index_no=df.columns.get_loc(col_name)
        indices.append(index_no)
    return indices

def build_model_SVC(df):
    """
            It builds a model using Support Vector regresion Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.svm import SVC
    model = SVC(probability=True)
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)
    # i, j=indices[0], indices[1]

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    df2=pd.DataFrame(clf.cv_results_)
    df2['mean_test_score']=df2['mean_test_score'] * 100
    enc_kernel=LabelEncoder().fit_transform(df2['param_kernel'])
    df2=df2.drop(['params', 'param_kernel'], axis=1)
    df2['param_kernel']=enc_kernel

    if gamma_sel != 0:
        enc_gamma=LabelEncoder().fit_transform(df2['param_gamma'])
        df2 = df2.drop(['param_gamma'], axis=1)
        df2['param_gamma']=enc_gamma

    df2=df2[['param_C', 'param_gamma', 'param_kernel', 'mean_test_score']].astype(int)

    fig=px.parallel_coordinates(df2, color="mean_test_score",
                            labels=dict(zip(list(df2.columns),
                                   list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),
                            color_continuous_scale=px.colors.diverging.Tealrose,
                            color_continuous_midpoint=27)

    st.header("Parallel Coordinates Plot")
    st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" % f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm=confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig=ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)

def build_model_Logistic(df):
    """
            It builds a model using Logistic Regresion Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    df2=pd.DataFrame(clf.cv_results_)
    df2['mean_test_score']=df2['mean_test_score'] * 100
    enc_penalty=LabelEncoder().fit_transform(df2['param_penalty'])
    df2=df2.drop(['params', 'param_penalty'], axis=1)
    df2['param_penalty']=enc_penalty

    df2=df2[['param_C', 'param_penalty', 'mean_test_score']].astype(int)

    fig=px.parallel_coordinates(df2, color="mean_test_score",
                            labels=dict(zip(list(df2.columns),
                                   list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),
                            color_continuous_scale=px.colors.diverging.Tealrose,
                            color_continuous_midpoint=27)

    st.header("Parallel Coordinates Plot")
    st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" % f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm=confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig=ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)

def build_model_KNeighbours(df):
    """
            It builds a model using K Neighbours Classification Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    df2=pd.DataFrame(clf.cv_results_)
    df2['mean_test_score']=df2['mean_test_score'] * 100
    enc_weights=LabelEncoder().fit_transform(df2['param_weights'])
    enc_algorithm=LabelEncoder().fit_transform(df2['param_algorithm'])
    df2=df2.drop(['params', 'param_weights', 'param_algorithm'], axis=1)
    df2['param_weights']=enc_weights
    df2['param_algorithm']=enc_algorithm

    df2=df2[['param_n_neighbors', 'param_leaf_size', 'param_algorithm', 'param_weights', 'mean_test_score']].astype(int)

    fig=px.parallel_coordinates(df2, color="mean_test_score",
                            labels=dict(zip(list(df2.columns),
                                   list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),
                            color_continuous_scale=px.colors.diverging.Tealrose,
                            color_continuous_midpoint=27)

    st.header("Parallel Coordinates Plot")
    st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" % f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm = confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig = ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)

def build_model_RandomForest(df):
    """
            It builds a model using Random Forest Classification Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    df2=pd.DataFrame(clf.cv_results_)
    df2['mean_test_score']=df2['mean_test_score'] * 100
    enc_criterion=LabelEncoder().fit_transform(df2['param_criterion'])
    enc_bootstrap=LabelEncoder().fit_transform(df2['param_bootstrap'])
    enc_max_features = LabelEncoder().fit_transform((df2['param_max_features']))
    df2 = df2.drop(['param_max_features'], axis=1)
    df2['param_max_features'] = enc_max_features

    if max_depth[0] is None:
        enc_max_depth=LabelEncoder().fit_transform((df2['param_max_depth']))
        df2=df2.drop(['param_max_depth'], axis=1)
        df2['param_max_depth']=enc_max_depth

    df2=df2.drop(['params', 'param_criterion', 'param_bootstrap'], axis=1)
    df2['param_criterion']=enc_criterion
    df2['param_bootstrap']=enc_bootstrap

    df2=df2[['param_n_estimators', 'param_criterion', 'param_max_depth', 'param_bootstrap'
             'param_min_samples_split', 'param_min_samples_leaf',  'mean_test_score']].astype(int)

    fig=px.parallel_coordinates(df2, color="mean_test_score",
                            labels=dict(zip(list(df2.columns),
                                   list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),
                            color_continuous_scale=px.colors.diverging.Tealrose,
                            color_continuous_midpoint=27)

    st.header("Parallel Coordinates Plot")
    st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" % f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm = confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig = ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)

def build_model_SGD(df):
    """
            It builds a model using Stocastic Gradient Descent Classification Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier()
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" % f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm = confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig = ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)

def build_model_DecisionTree(df):
    """
            It builds a model using Decision Tree Classification Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    model = DecisionTreeClassifier()
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    df2=pd.DataFrame(clf.cv_results_)
    df2['mean_test_score']=df2['mean_test_score'] * 100
    enc_criterion=LabelEncoder().fit_transform(df2['param_criterion'])
    enc_splitter=LabelEncoder().fit_transform(df2['param_splitter'])
    enc_max_features=LabelEncoder().fit_transform((df2['param_max_features']))
    df2=df2.drop(['param_max_features'], axis=1)
    df2['param_max_features']=enc_max_features

    if max_depth[0] is None:
        enc_max_depth=LabelEncoder().fit_transform((df2['param_max_depth']))
        df2=df2.drop(['param_max_depth'], axis=1)
        df2['param_max_depth']=enc_max_depth

    df2=df2.drop(['params', 'param_criterion', 'param_splitter'], axis=1)
    df2['param_criterion']=enc_criterion
    df2['param_splitter']=enc_splitter

    df2=df2[['param_criterion', 'param_max_depth', 'param_splitter', 'param_min_samples_split',
             'param_min_samples_leaf', 'mean_test_score']].astype(int)

    fig=px.parallel_coordinates(df2, color="mean_test_score",
                                labels=dict(zip(list(df2.columns),
                                                list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                color_continuous_midpoint=27)

    st.header("Parallel Coordinates Plot")
    st.plotly_chart(fig)

    st.header("Decision Tree")
    import matplotlib.pyplot as plt
    dtree = DecisionTreeClassifier(**clf.best_params_)
    dtree.fit(X_train, Y_train)
    fn=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    cn=['setosa', 'versicolor', 'virginica']
    fig, axes=plt.subplots(nrows=1, ncols=1, dpi=300)
    plot_tree(dtree,
               feature_names = fn,
               class_names=cn,
               filled = True)
    st.pyplot(fig)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" % f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm = confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig = ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)

def build_model_GradientBoosting(df):
    """
            It builds a model using Gradient Boosting Classification Algorithm.
            Takes input from streamlit web interface and use those inputs for building the model.
            Used GridSearchCV for Hyperparameter Tunning.
            Ploting the result using Plotly Framework.

            :param df: dataframe containing features and labels.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    clf=GridSearchCV(model, param_grid)

    ind_var_length = len(ind_var)
    indices=getIndex(df)

    st.header("Decision Boundaries for different classes")
    for i in range(0, ind_var_length - 1):
        for j in range(i + 1, ind_var_length):
            k, l=indices[i], indices[j]

            x=df.iloc[:, [k, l]].values  # only for 2 features
            y=df.iloc[:, -1].values  # Selecting the last column as Y

            X_train, X_test, Y_train, Y_test0=train_test_split(x, y, test_size=split_size, shuffle=True)

            clf.fit(X_train, Y_train)

            fig=DecisionBoundary(clf, x, y)
            fig.update_xaxes(title=ind_var[i])
            fig.update_yaxes(title=ind_var[j])
            st.plotly_chart(fig)

    X= df[ind_var]
    Y= df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=split_size)
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    df2=pd.DataFrame(clf.cv_results_)
    df2 = df2.dropna()
    df2['mean_test_score']=df2['mean_test_score'] * 100
    enc_loss=LabelEncoder().fit_transform(df2['param_loss'])
    df2['param_loss']=enc_loss

    df2=df2[['param_n_estimators', 'param_loss', 'param_max_depth', 'param_learning_rate',
             'param_subsample', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']].astype(int)

    fig=px.parallel_coordinates(df2, color="mean_test_score",
                            labels=dict(zip(list(df2.columns),
                                   list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),
                            color_continuous_scale=px.colors.diverging.Tealrose,
                            color_continuous_midpoint=27)

    st.header("Parallel Coordinates Plot")
    st.plotly_chart(fig)

    st.write("The best parameters are %s with a score of %0.2f "
             % (clf.best_params_, clf.best_score_*100))

    st.subheader('Model Parameters')
    st.write(clf.get_params())

    st.subheader('Model Performance')
    st.write('F1 Score (1 is best, 0 is worst):')
    st.info("%0.2f" %f1_score(Y_test, Y_pred_test, average= average))

    y_scores=clf.predict_proba(X_test)

    if criterion == 'Recall':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Precision':
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
    if criterion == 'Accuracy':
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
    if criterion == 'ROC AUC':
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class=multi_class) * 100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores) * 100))
    if criterion == 'All':
        st.write('Recall Score')
        st.info("%0.2f" % (recall_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Precision Score:')
        st.info("%0.2f" % (precision_score(Y_test, Y_pred_test, average= average)*100))
        st.write('Accuracy Score:')
        st.info("%0.2f" % (accuracy_score(Y_test, Y_pred_test)*100))
        st.write('ROC AUC Score:')
        if check == 'True':
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores, multi_class= multi_class)*100))
        else:
            st.info("%0.2f" % (roc_auc_score(Y_test, y_scores,) * 100))

    st.header("Confusion Matrix")
    cm = confusion_matrix(Y_test, Y_pred_test)
    z_text=[[str(y) for y in x] for x in cm]
    d=list(clf.classes_)
    fig = ConfusionMatrix(cm, z_text, d)
    st.plotly_chart(fig)

    # One hot encode the labels in order to plot them
    y_onehot=pd.get_dummies(Y_test, columns=clf.classes_)
    fig = ROC_AUC(y_scores, y_onehot)
    st.header("ROC curve (receiver operating characteristic curve)")
    st.plotly_chart(fig)






# Page Layout ( Streamlit web Interface )
st.set_page_config(page_title="Classification Model Builder")

st.write("""
# Classification Model Builder
""")

# Sidebar ..............................................

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.header("Dataset Configuration")
split_size = st.sidebar.slider('Data Split Ratio (training set)', 10,90,80,5)

st.sidebar.header("Select Classifier")
reg = st.sidebar.selectbox("Choose Classification Algorithm", options=['Logistic Regression (Binary)', 'SVC',
                            'K Neighbours Classification', 'Random Forest Classification', 'Decision Tree Classification',
                            'Gradient Boosting', 'SGD Classification'])


if reg == 'SVC':

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var = st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average = st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)', options = ['micro', 'macro', 'binary', 'weighted'])

    check = st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)', options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Support Vector Classification")
    st.sidebar.subheader("Kernel")
    kernel = st.sidebar.selectbox("Enter from the options", options=['All', 'linear', 'rbf', 'poly'])

    numbers = st.sidebar.text_input("Enter values for 'c'. (Separate values with ,)", value='1.0')
    C = collect_numbers(numbers)

    st.sidebar.subheader('Select the value for gamma from the options or provide your custom values')
    gamma_sel = st.sidebar.selectbox("Enter from the options", options=['', 'All', 'auto', 'scale'])
    st.sidebar.header("OR")
    numbers = st.sidebar.text_input("Enter values for 'gamma'. (Separate values with ,)")
    gamma_float = collect_numbers(numbers)

    if kernel == 'All':
        kernel = ['linear', 'rbf', 'poly']
    else:
        kernel = [kernel]

    if len(gamma_sel) == 0:
        gamma = gamma_float
    elif gamma_sel == 'All':
        gamma = ['auto', 'scale']
    else:
        gamma = [gamma_sel]

    param_grid = dict(kernel = kernel, gamma = gamma, C = C)

if reg == 'Logistic Regression (Binary)':

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var=st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)',
                                       options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average=st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)',
                                     options=['micro', 'macro', 'binary', 'weighted'])

    check=st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)',
                                             options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Logistic Regression")
    penalty=['l2']

    numbers=st.sidebar.text_input("Enter values for 'C'. (Separate values with ,)", value='1.0')
    C=collect_numbers(numbers)

    numbers=st.sidebar.text_input("Enter Maximum iterations. (Separate values with ,)", value='100')
    max_iter =collect_numbers_int(numbers)


    param_grid=dict(penalty = penalty, C=C, max_iter = max_iter)

if reg == 'K Neighbours Classification':

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var=st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)',
                                       options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average=st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)',
                                     options=['micro', 'macro', 'binary', 'weighted'])

    check=st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)',
                                             options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for K Neighbours Classification")

    numbers=st.sidebar.text_input("Number of Neighbours (Separate values with ,)", value='5')
    n_neighbors = collect_numbers_int(numbers)

    weights=st.sidebar.selectbox('Weights', options=['All', 'uniform', 'distance'])

    algorithm=st.sidebar.selectbox('Algorithm', options=['All', 'auto', 'ball_tree', 'kd_tree', 'brute'])

    numbers=st.sidebar.text_input("Leaf Size (Separate values with ,)", value='30')
    leaf_size=collect_numbers_int(numbers)

    if weights == 'All':
        weights = ['uniform', 'distance']
    else:
        weights = [weights]

    if algorithm == 'All':
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    else:
        algorithm = [algorithm]

    param_grid=dict(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm, leaf_size = leaf_size)

if reg == 'Random Forest Classification':

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var=st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)',
                                       options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average=st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)',
                                     options=['micro', 'macro', 'binary', 'weighted'])

    check=st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)',
                                             options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Random Forest Classification")

    numbers=st.sidebar.text_input("Number of Estimators (Separate values with ,)", value='100')
    n_estimators = collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("Maximum Depth of tree (Separate values with ,)", value='None')
    if numbers == 'None':
        max_depth = [None]
    else:
        max_depth = collect_numbers_int(numbers)

    ct=st.sidebar.selectbox('Criterion', options=['All', 'gini', 'entropy'])

    max_features=st.sidebar.selectbox('Maximum Features', options=['All', 'auto', 'sqrt', 'log2'])

    numbers=st.sidebar.text_input("Minimum Samples Split (Separate values with ,)", value='2')
    min_samples_split=collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("Minimum Samples leaf (Separate values with ,)", value='1')
    min_samples_leaf=collect_numbers_int(numbers)

    bootstrap = st.sidebar.selectbox("Bootstrap", options=['Both', 'True', 'False'])

    if bootstrap == 'Both':
        bootstrap = ['True', 'False']
    else:
        bootstrap = [bootstrap]

    if ct == 'All':
        ct = ['gini', 'entropy']
    else:
        ct = [ct]

    if max_features == 'All':
        max_features = ['auto', 'sqrt', 'log2']
    else:
        max_features = [max_features]


    param_grid=dict(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split,
                    max_features = max_features, max_depth = max_depth, criterion = ct, bootstrap = bootstrap)

if reg == 'SGD Classification':

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var=st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)',
                                       options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average=st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)',
                                     options=['micro', 'macro', 'binary', 'weighted'])

    check=st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)',
                                             options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Stocastic Gradient Descent Classification")

    numbers=st.sidebar.text_input("Maximum Iteration (Separate values with ,)", value='1000')
    max_iter = collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("alpha (Separate values with ,)", value='0.0001')
    alpha=collect_numbers(numbers)

    loss=st.sidebar.selectbox('Loss', options=['All', 'hinge', 'log', 'modified_huber', 'perceptron'])

    learning_rate = st.sidebar.selectbox('Learning Rate', options=['All', 'constant', 'optimal', 'invscaling', 'adaptive'])

    numbers=st.sidebar.text_input("L1 ration (Separate values with ,)", value='0.15')
    l1_ratio=collect_numbers(numbers)

    early_stopping=st.sidebar.selectbox('Early Stopping', options=['All', 'False', 'True'])

    if loss == 'All':
        loss = ['hinge', 'log', 'modified_huber', 'perceptron']
    else:
        loss = [loss]

    if learning_rate == 'All':
        learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
    else:
        learning_rate = [learning_rate]

    if early_stopping == 'All':
        early_stopping = [True, False]
    else:
        early_stopping = [early_stopping]

    param_grid=dict(loss = loss, early_stopping = early_stopping, learning_rate = learning_rate, l1_ratio = l1_ratio,
                    max_iter = max_iter, alpha=alpha)

if reg == 'Decision Tree Classification':

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var=st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)',
                                       options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average=st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)',
                                     options=['micro', 'macro', 'binary', 'weighted'])

    check=st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)',
                                             options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Decision Tree Classification")

    numbers=st.sidebar.text_input("Maximum Depth of tree (Separate values with ,)", value='None')
    if numbers == 'None':
        max_depth = [None]
    else:
        max_depth = collect_numbers_int(numbers)

    ct=st.sidebar.selectbox('Criterion', options=['All', 'gini', 'entropy'])

    splitter=st.sidebar.selectbox('Criterion', options=['All', 'best', 'random'])

    max_features=st.sidebar.selectbox('Maximum Features', options=['All', 'auto', 'sqrt', 'log2'])

    numbers=st.sidebar.text_input("Minimum Samples Split (Separate values with ,)", value='2')
    min_samples_split=collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("Minimum Samples leaf (Separate values with ,)", value='1')
    min_samples_leaf=collect_numbers_int(numbers)

    if splitter == 'All':
        splitter = ['best', 'random']
    else:
        splitter = [splitter]

    if ct == 'All':
        ct = ['gini', 'entropy']
    else:
        ct = [ct]

    if max_features == 'All':
        max_features = ['auto', 'sqrt', 'log2']
    else:
        max_features = [max_features]


    param_grid=dict(splitter = splitter, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split,
                    max_features = max_features, max_depth = max_depth, criterion = ct)

if reg == 'Gradient Boosting':

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
    else:
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

    st.sidebar.subheader('Variable Configuration')
    ind_var=st.sidebar.multiselect('Choose Independent Variables', options=df.columns)

    st.sidebar.write('---')
    criterion=st.sidebar.selectbox('Performance measure (criterion)',
                                       options=['All', 'Recall', 'Precision', 'Accuracy', 'ROC AUC'])
    average=st.sidebar.selectbox('Average for F1 Score (Choose Binary for 2 Class)',
                                     options=['micro', 'macro', 'binary', 'weighted'])

    check=st.sidebar.selectbox('Multi Classification', options=['True', 'False'])
    if check == 'True':
        multi_class=st.sidebar.selectbox('If data is multi class then choose One vs One(ovo) OR One vs Rest(ovr)',
                                             options=['ovo', 'ovr'])

    st.sidebar.write('---')
    st.sidebar.subheader("Hyperparameters for Gradient Boosting Classification")

    numbers=st.sidebar.text_input("Learning Rate (Separate values with ,)", value='0.1')
    learning_rate = collect_numbers(numbers)

    numbers=st.sidebar.text_input("Number of Estimators (Separate values with ,)", value='100')
    n_estimators=collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("Subsamples (Separate values with ,)", value='1.0')
    subsample=collect_numbers(numbers)

    loss=st.sidebar.selectbox('Loss', options=['All', 'deviance', 'exponential'])

    numbers=st.sidebar.text_input("Minimum Samples Split (Separate values with ,)", value='2')
    min_samples_split=collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("Minimum Samples leaf (Separate values with ,)", value='1')
    min_samples_leaf=collect_numbers_int(numbers)

    numbers=st.sidebar.text_input("Maximum Depth of tree (Separate values with ,)", value='3')
    max_depth = collect_numbers_int(numbers)

    if loss == 'All':
        loss = ['deviance', 'exponential']
    else:
        loss = [loss]

    param_grid=dict(loss=loss, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    subsample= subsample, n_estimators=n_estimators, learning_rate = learning_rate)



# main Body ...............................................................................................

st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    if reg == 'SVC':
        build_model_SVC(df)
    if reg == 'Logistic Regression (Binary)':
        build_model_Logistic(df)
    if reg == 'K Neighbours Classification':
        build_model_KNeighbours(df)
    if reg == 'Random Forest Classification':
        build_model_RandomForest(df)
    if reg == 'SGD Classification':
        build_model_SGD(df)
    if reg == 'Decision Tree Classification':
        build_model_DecisionTree(df)
    if reg == 'Gradient Boosting':
        build_model_GradientBoosting(df)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        iris=load_iris()
        X=pd.DataFrame(iris.data, columns=iris.feature_names)
        Y=pd.Series(iris.target, name='Species')
        df=pd.concat([X, Y], axis=1)

        st.markdown('The **IRIS** dataset is used as the example.')
        st.write(df.head(5))

        if reg == 'SVC':
            build_model_SVC(df)
        if reg == 'Logistic Regression (Binary)':
            build_model_Logistic(df)
        if reg == 'K Neighbours Classification':
            build_model_KNeighbours(df)
        if reg == 'Random Forest Classification':
            build_model_RandomForest(df)
        if reg == 'SGD Classification':
            build_model_SGD(df)
        if reg == 'Decision Tree Classification':
            build_model_DecisionTree(df)
        if reg == 'Gradient Boosting':
            build_model_GradientBoosting(df)