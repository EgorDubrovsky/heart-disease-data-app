import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from messages import msg
import dalex as dx
import pickle

# Useful global stuff
DATA_PATH = './data/heart_disease.csv'
DIAGNOSIS_DICT = {0: 'hd0',
                  1: 'hd1',
                  2: 'hd2',
                  3: 'hd3',
                  4: 'hd4'}
MODELS_SEED = 123

assert msg.FEATURES_DESC.keys() == msg.FEATURES_VALS.keys()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data load
df_raw = pd.read_csv(DATA_PATH)  # This has all rows
df = df_raw.dropna()             # This has no rows with missing values

# Header
st.title(msg.MAIN_TITLE)
st.write(msg.APP_DESCRIPTION)

# Tabs setup
tab_data, tab_model, tab_xai, tab_fairness = st.tabs(['Exploring Data', 'Building Models', 'Explaining Algorithms', 'Evaluating Fairness'])


# ================
# === SIDE BAR ===
# ================

sidebar = st.sidebar
sidebar.header(msg.FILTER_TITLE)

# Age slider
age = sidebar.slider('Age', int(df['age'].min()), int(df['age'].max()), 
                            (int(df['age'].min()), int(df['age'].max())))

# Sex selectbox
sex = sidebar.selectbox('Sex', ('Both', 'Male', 'Female'))
    
# Diagnosis multiselect
diagnosis = sidebar.multiselect('Diagnosis', df['num'].unique(), df['num'].unique(),
                                    format_func=lambda x: DIAGNOSIS_DICT[round(x)])

# Filtering the dataframe
sex_set = {0} if sex == 'Female' else {1} if sex == 'Male' else {0, 1}
filtered_df = df[(df['age'] >= age[0]) & (df['age'] <= age[1]) & (df['num'].isin(diagnosis)) \
                 & df['sex'].isin(sex_set)]

# Count the filtered data
sidebar.write('Total filtered entries: ' + str(filtered_df.shape[0]))
sidebar.write('Fraction of filtered entries: ' + str(round(filtered_df.shape[0] / df.shape[0] * 100, 2)) + '%')

# Filter subsection ends
sidebar.write(msg.SEPARATOR)

# Dataset features info
sidebar.header(msg.FEATURE_TITLE)
sidebar.write(msg.SELECT_FEATURE_MSG)
selectbox_choices = list(df.columns)
selectbox_choices.append('this_is_a_missing_feature')
trivia_target = sidebar.selectbox("Select Feature", selectbox_choices)
if trivia_target in msg.FEATURES_DESC.keys() and trivia_target in msg.FEATURES_VALS.keys():
    sidebar.write('🤓 Info on *' + trivia_target + '* 🤓')
    sidebar.write('***Description***: ' + msg.FEATURES_DESC[trivia_target])
    sidebar.write('***Possible values***: ' + msg.FEATURES_VALS[trivia_target])
else:
    sidebar.write(msg.MISSING_FEATURE_ERROR)

# Dataset features subsection ends
sidebar.write(msg.SEPARATOR)

# Dataset general info
sidebar.header(msg.DATA_DESCRIPTION_TITLE)
for line in msg.DATASET_DESCRIPTION:
    sidebar.write(line)


# ================
# === DATA TAB ===
# ================



with tab_data:

    # Full dataframe
    st.write(msg.FULL_DATASET_TITLE)
    st.write(msg.FULL_DATASET_MSG)
    st.dataframe(df_raw)

    # Filtered dataframe
    st.write(msg.FILTERED_DATASET_TITLE)
    st.write(msg.FITERED_DATASET_MSG)
    
    if diagnosis:
        st.dataframe(filtered_df)
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

    # Individual distributions
    HIST_REQUIRED = ('age', 'trestbps', 'chol', 'thalach', 'oldpeak')
    st.write(msg.FEATURE_DISTRIBUTION_TITLE)
    st.write(msg.FEATURE_DISTRIBUTION_MSG)
    if diagnosis:
        feature_name = st.selectbox("Select Feature", df.columns)
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of feature ' + feature_name)
        if feature_name == 'PhysHlth' or feature_name == 'MentHlth':
            plt.xticks(rotation=90)
        if feature_name in HIST_REQUIRED:
            ax.hist(filtered_df[feature_name], bins=30, color='skyblue', edgecolor='black')
        else:
            ax.bar(np.vectorize(lambda x: str(int(x)))(np.unique(filtered_df[feature_name])), 
                np.unique(filtered_df[feature_name], return_counts=True)[1], edgecolor='black',
                color='skyblue')
        st.pyplot(fig)
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

# TODO: add correlations

# ==================
# === MODELS TAB ===
# ==================



# ===============
# === XAI TAB ===
# ===============



# ====================
# === FAIRNESS TAB ===
# ====================
