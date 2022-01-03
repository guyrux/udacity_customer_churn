# library doc string
'''This module has all custom methods to use in this project.
'''

from configparser import ConfigParser

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_roc_curve, classification_report


config = ConfigParser()
config.read('settings.ini')

EXTERNAL_DATA_PATH = config.get('FOLDERS', 'external_data')
MODEL_PATH = config.get('FOLDERS', 'models')
EDA_PATH = config.get('FOLDERS', 'eda')
RESULT_PATH = config.get('FOLDERS', 'results')
RANDON_STATE = 42

def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    figsize = (20, 5)
    font_size = 20
    lst_category = ['Churn', 'Customer_Age']

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    for item in lst_category:
        plt.figure(figsize=figsize)
        plt.title(f'Histogram {item}', size=font_size)
        df[item].hist()
        plt.savefig(f'histogram_{item.lower()}.png', bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.title('Distplot Total_Trans_Ct', size=font_size)
    sns.histplot(df['Total_Trans_Ct'], kde=True)
    plt.savefig('distplot Total_Trans_Ct.png', bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.title('Heatmap', size=font_size)
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=1)
    plt.savefig('heatmap.png', bbox_inches='tight')


def encoder_helper(df, category_lst, response: str = None) -> None:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for
    '''
    df_temp = df.copy()
    df_temp['Churn'] = df_temp['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    for category in category_lst:
        temp_groups = df_temp.groupby(category).mean()['Churn']

        temp_lst = [temp_groups.loc[val] for val in df_temp[category]]
        df_temp[f'{category}_Churn'] = temp_lst

    return df_temp


def perform_feature_engineering(df, response: str = None) -> pd.DataFrame:
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    X = pd.DataFrame()
    y = df['Churn']

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf
):
    '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

        output:
            None
        '''
    lst_model = ['Logistic model', 'Random forest']

    for model in lst_model:
        if model == 'Logistic model':
            y_test_preds = y_test_preds_lr
            y_train_preds = y_train_preds_lr
        else:
            y_test_preds = y_test_preds_rf
            y_train_preds = y_train_preds_rf

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str(f'{model} Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model} Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off');

        plt.savefig(f'classification_reports_{model.lower().join("_")}.png', bbox_inches='tight')


def feature_importance_plot(model: str = 'lr', X_data: pd.DataFrame = None, output_pth: str = RESULT_PATH) -> None:
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''
    rfc_model = joblib.load('../models/rfc_model.pkl')
    lr_model = joblib.load('../models/logistic_model.pkl')

    # Calculate feature importances
    if model == 'lr':
        importances = lr_model.coef_[0]
    else:
        importances = rfc_model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance", size=20)
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(f'{output_pth}feature_importance_{model}.png', bbox_inches='tight')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    # grid search
    font_size = 20

    rfc = RandomForestClassifier(random_state=RANDON_STATE, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    joblib.dump(cv_rfc.best_estimator_, MODEL_PATH + 'rfc_model.pkl')
    joblib.dump(lrc, MODEL_PATH + 'logistic_model.pkl')

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.title('ROC', size=font_size)
    plt.savefig('roc.png', bbox_inches='tight')

