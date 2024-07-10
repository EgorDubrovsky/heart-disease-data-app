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
from sklearn.metrics import classification_report
import shap
from explainerdashboard import ClassifierExplainer

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
df_raw['num'] = df_raw['num'].apply(lambda x: 1 if x > 0 else x)
df = df_raw.dropna()  # This has no rows with missing values

# Header
st.title(msg.MAIN_TITLE)
st.write(msg.APP_DESCRIPTION)

# Tabs setup
tab_data, tab_model, tab_xai, tab_fairness = st.tabs(
    ['Exploring Data', 'Building Models', 'Explaining Algorithms', 'Evaluating Fairness'])

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

X_new = df.drop("num", axis=1)
y_new = df["num"]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=123,
                                                                    stratify=y_new)


@st.cache_resource
def train_rf_model(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return rf_model, pd.DataFrame(report).transpose()


@st.cache_resource
def train_lr_model(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(max_iter=1000, random_state=123)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return lr_model, pd.DataFrame(report).transpose()


# Train the models on the new dataset
rf_model, rf_report_df = train_rf_model(X_train_new, y_train_new, X_test_new, y_test_new)
lr_model, lr_report_df = train_lr_model(X_train_new, y_train_new, X_test_new, y_test_new)

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
    CONTINUOUS = ('age', 'trestbps', 'chol', 'thalach', 'oldpeak')  # These features will need a histogram
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
with tab_model:
    st.header("Building And Explaining Models")

    # Model selection
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"], key="model_choice")

    if model_choice == "Random Forest":
        st.subheader("Random Forest Model")
        st.write("Results of the model training")
        st.write(rf_report_df)

    elif model_choice == "Logistic Regression":
        st.subheader("Logistic Regression Model")
        st.write("Results of the model training")
        st.write(lr_report_df)


# ===============
# === XAI TAB ===
# ===============
with tab_xai:
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"], key="model_choice_xai")
    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = lr_model

    st.header("Model Explanations")
    explainer = ClassifierExplainer(model, X_test_new, y_test_new, shap_kwargs=dict(approximate=True))

    st.subheader("Feature Importance")
    fi = explainer.get_importances_df()
    st.bar_chart(fi.set_index('Feature')['MEAN_ABS_SHAP'])

    st.subheader("SHAP Summary Plot")
    shap_values = explainer.get_shap_values_df()
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values.values, X_test_new, show=False)
    st.pyplot(fig)

    st.write("""
    - **Feature Importance** helps identify which features are most influential overall.
    - **SHAP Summary Plot** provides a global interpretation of feature impacts.
    """)

# ====================
# === FAIRNESS TAB ===
# ====================
