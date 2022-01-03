from configparser import ConfigParser
import logging
import os

import pandas as pd

from churn_library import (
    import_data, encoder_helper, perform_feature_engineering, train_models
)

# import churn_library_solution as cls

config = ConfigParser()
config.read('settings.ini')

EXTERNAL_DATA_PATH = config.get('FOLDERS', 'external_data')
MODEL_PATH = config.get('FOLDERS', 'models')
EDA_PATH = config.get('FOLDERS', 'eda')
RESULT_PATH = config.get('FOLDERS', 'results')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)

category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(EXTERNAL_DATA_PATH + 'bank_data.csv')
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


# def test_eda(perform_eda):
#     '''
#     test perform eda function
#     '''
#     pass


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        df = import_data(EXTERNAL_DATA_PATH + 'bank_data.csv')
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        df_output = encoder_helper(df, category_lst)
        assert len(df_output.columns) == len(df.columns) + len(category_lst) + 1
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: The encoder do NOT returned the correct number of columns.")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    df = import_data("./data/external/bank_data.csv")
    logging.info("Testing import_data: SUCCESS")

    df_output = encoder_helper(df, category_lst)
    logging.info("Testing encoder_helper: SUCCESS")

    X_train, X_test, y_train, y_test = perform_feature_engineering(df_output)
    logging.info("Testing perform_feature_engineering: SUCCESS")

    assert len(X_train.columns) == 19
    assert len(X_test.columns) == 19
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_train_models():
    '''
    test train_models
    '''

    df = import_data("./data/external/bank_data.csv")
    logging.info("Testing import_data: SUCCESS")

    df_output = encoder_helper(df, category_lst)
    logging.info("Testing encoder_helper: SUCCESS")

    X_train, X_test, y_train, y_test = perform_feature_engineering(df_output)
    logging.info("Testing perform_feature_engineering: SUCCESS")

    train_models(X_train, X_test, y_train, y_test)

    lst_model = ['logistic_model.pkl', 'rfc_model.pkl']

    for model in lst_model:
        assert os.path.isfile(MODEL_PATH + model)
        logging.info(f'Testing model {model}: SUCCESS.')

    # TODO: Check if images were saved. São 5 imagens.
    # lst_images = [
    #     'roc.png', 'classification_reports_rf.png', 'classification_reports_lr.png'
    # ]

    # for image in lst_images:
    #     assert os.path.isfile(MODEL_PATH + image)
    #     logging.info(f'Testing results - {image}: SUCCESS.')


if __name__ == "__main__":
    test_import()
    test_train_models()
