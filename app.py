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
from viz import visualize
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
    
    if filtered_df.shape[0]:
        st.dataframe(filtered_df)
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

    # Individual distributions
    CONTINUOUS = ('age', 'trestbps', 'chol', 'thalach', 'oldpeak') # These features will need a histogram
    st.write(msg.FEATURE_DISTRIBUTION_TITLE)
    st.write(msg.FEATURE_DISTRIBUTION_MSG)

    if filtered_df.shape[0]:
        feature_name = st.selectbox("Select Feature", df.columns)
        if feature_name in CONTINUOUS:
            st.pyplot(visualize().continuous_distr(filtered_df, feature_name))
        else:
            st.pyplot(visualize().categorical_distr(filtered_df, feature_name))
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

    # Mutual distributions
    st.write(msg.FEATURE_CORRELATION_TITLE)
    st.write(msg.FEATURE_CORRELATION_MSG)
    if filtered_df.shape[0]:
        feature1_name = st.selectbox("Select Feature 1", df.columns, index=0)
        feature2_name = st.selectbox("Select Feature 2", df.columns, index=1)

        # If names are equal - just put out the marginal distribution
        if feature1_name == feature2_name: 
            if feature1_name in CONTINUOUS:
                st.pyplot(visualize().continuous_distr(filtered_df, feature1_name))
            else:
                st.pyplot(visualize().categorical_distr(filtered_df, feature1_name))
        # If names are not equal - several options
        else:
            # Both features continuous -> show scatterplot
            if feature1_name in CONTINUOUS and feature2_name in CONTINUOUS:
                st.pyplot(visualize().scatterplot(filtered_df, feature1_name, feature2_name))
            # One continuous, one categorical -> show boxplot
            elif feature1_name in CONTINUOUS:
                st.pyplot(visualize().boxplot(filtered_df, feature1_name, feature2_name))
            elif feature2_name in CONTINUOUS:
                st.pyplot(visualize().boxplot(filtered_df, feature2_name, feature1_name))
            # Both features categorical -> show heatmap?
            else:
                st.pyplot(visualize().heatmap(filtered_df, feature1_name, feature2_name))
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

# ==================
# === MODELS TAB ===
# ==================



# ===============
# === XAI TAB ===
# ===============



# ====================
# === FAIRNESS TAB ===
# ====================
