'''This is the main file from thie project.
    Using streamlit library it is possible to access the churn library created to this project.
'''

from configparser import ConfigParser
import glob

import streamlit as st

import churn_library as cl

config = ConfigParser()
config.read('settings.ini')

EXTERNAL_DATA_PATH = config.get('FOLDERS', 'external_data')
INTERIM_DATA_PATH = config.get('FOLDERS', 'interim_data')
MODEL_PATH = config.get('FOLDERS', 'models')
EDA_PATH = config.get('FOLDERS', 'eda')
RESULT_PATH = config.get('FOLDERS', 'results')
CAT_LST = config.get('VARIABLES', 'cat_columns').split(', ')
QUANT_LST = config.get('VARIABLES', 'quant_columns').split(', ')
RANDON_STATE = 42

category_lst = [
    'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'
]

st.title("Udacity's customer churn project")

data_load_state = st.text('Loading data...')
data = cl.import_data('./data/external/bank_data.csv')
data_load_state.text("Done! Data loaded.")

# Show/hide dataframe
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(data)

# Show/hide EDA
if st.checkbox('Show EDA'):
    st.subheader('EDA - Exploratory data analysis')
    cl.perform_eda(data)
    lst_file = glob.glob(EDA_PATH + '*.png')

    # Show all png files from EDA folder
    st.subheader('Categorical plots:')
    for image in [file for file in lst_file if 'proportion' in file]:
        st.image(image)

    st.subheader('Quantitative plots:')
    for image in [file for file in lst_file if 'distplot' in file]:
        st.image(image)

    st.subheader('Bivariate plot:')
    for image in [file for file in lst_file if 'heatmap' in file]:
        st.image(image)

# Show/hide model prediction
if st.checkbox('Show churn prediction'):
    st.subheader('Churn predictioon')
    train_state = st.text('Training model...')

    lst_model_files = glob.glob(MODEL_PATH + '*.pkl')

    if len(lst_model_files) < 2:
        data_temp = cl.encoder_helper(data, category_lst)
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(data_temp)
        cl.train_models(X_train, X_test, y_train, y_test)

    lst_file = glob.glob(RESULT_PATH + '*.png')

    # Show all png files from result folder
    for file in lst_file:
        st.image(file)

    train_state = st.text('Model trained and results are ready.')
