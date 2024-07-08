class msg():
    FORMULA_TOKEN = 'FORMULA'
    # Headers
    MAIN_TITLE = '🚑 Heart Disease Factors Data App'
    FILTER_TITLE = '🔬 Filters'
    FEATURE_TITLE = '📊 Features Information'
    DATA_DESCRIPTION_TITLE = '📜 Dataset Description'
    FULL_DATASET_TITLE = '### Cleveland Heart Disease Dataset (full)'
    FILTERED_DATASET_TITLE = '### Cleveland Heart Disease Dataset (filtered)'
    FEATURE_DISTRIBUTION_TITLE = '### Individual Feature Distributions (filtered)'
    MODEL_DUMMY_TITLE = '### Dummy Model'
    PERFOMANCE_DUMMY_TITLE = '#### 🐌 Dummy Model Perfomance'
    MODEL_LOGREG_TITLE = '### Simple Logistic Regression'
    MODEL_FOREST_TITLE = '### Random Forest'
    MODEL_BOOST_TITLE = '### Gradient Boosting'
    FAIRNESS_LOGREG_TITLE = '### Fairness Metrics for Logistic Regression'
    FAIRNESS_MOD_TITLE = '###  Modified Fairness Metrics for Logistic Regression'
    FAIRNESS_DISCUSS_TITLE = '### Discussion'
    XAI_LOGREG_TITLE = '### Explaining Logistic Regression'
    XAI_BOOST_TITLE = '### Explaining Gradient Boosting'
    XAI_BOOST_INDIVIDUAL_TITLE = '#### 🤒❓ Explaining Individual Samples'
    XAI_BOOST_GLOBAL_TITLE = '#### 👨‍👩‍👧‍👦❓ Global Explanations'

    # Text snippets
    SELECT_FEATURE_MSG = 'Select the feature you would like to know more about!'
    FULL_DATASET_MSG = 'Here you can see the entire dataset, unaffected by filters.'
    FITERED_DATASET_MSG = 'Adjust the filters on the sidebar to see how the dataset changes.'
    FEATURE_DISTRIBUTION_MSG = 'Select the feature to see their distribution. You can also interact with the filters to get more specific distributions.'
    LOGREG_TRAIN_MSG = '👇 Click the button to train a logistic regression model'
    FOREST_TRAIN_MSG = '👇 Click the button to train a random forest model'
    BOOST_TRAIN_MSG = '👇 Click the button to train a gradient boosting model'
    FOREST_HYPERPARAMS_MSG = '🔧 You can adjust the model\'s hyperparameters with the options below'
    FAIRNESS_LOGREG_MSG = 'First off, let\'s see the fairness metrics for the simple log resgression model.'
    FAIRNESS_MOD_MSG = 'To mitigate differences in metrics, we can try to remove the *Sex* feature from the training dataset. Here are the results.'

    # Longtexts
    APP_DESCRIPTION = 'This app provides an outlook on Cleveland Heart Disease dataset. \
                       You can examine surface-level data analysis, as well as predicitons of \
                       simple machine learning models and their fairness.\n \n For more information:\n \
                       https://archive.ics.uci.edu/dataset/45/heart+disease'
    DATASET_DESCRIPTION = ['*Created by*:',
                           '*Created in*:',
                           '*Purpose of creation*:',
                           '*Data collection method*:']
    MODELS_DESCRIPTION = ['On this tab you can explore the work of diffrent classification models which predict \
                          the person\'s health status.', '❗**IMPORTANT**❗: for simplicity, we combine labels 1 \
                          (prediabetes) and 2 (diabetes) together. This allows for the simpler binary \
                          classification and more interpretable metrics.', '❗**IMPORTANT**❗: data will **not** \
                          be filtered for training the models.'
                          ]
    MODEL_DUMMY_DESCRIPTION = 'Let\'s consider a trivial model which *labels all people as healthy*. \
                               While being useless on its own, it may be useful to compare its stats \
                               to smarter models (to see if they really are as smart as we think...)'
    MODEL_LOGREG_DESCRIPTION = 'Logistic regression is a simple classifier and one of the most iconic and used machine learning algorithms. \
                                Let\'s see how it tries to build predictions for our dataset.'
    MODEL_FOREST_DESCRIPTION = 'Random forest is a more advanced classification model which consists of many smaller models called decision trees. \
                                Will it be any better than the more shallow models?'
    MODEL_BOOST_DESCRIPTION = 'Finally, we can try gradient boosting - arguably the best-performing classification model (excluding \
                               neural networks, of course). Let\'s try feeding our dataset to it!'
    FAIRNESS_DESCRIPTION = ['On this tab you can explore different metrics associated with gender fairness, \
                            and see how fairly our models fare.', 'Yes, that was a pun.', 'No, I\'m not sorry.'
                           ]
    FAIRNESS_DISCUSSION = 'In reality, it makes little to no sense to assess the Group Fairness metric, since the frequency of diabetes \
                           is inherently different in men and women. On th other hand, metrics such as Predictive Parity or FPR Balance \
                           are useful, since they show us if the model makes the same amount of mistakes on men and women.'
    XAI_DESCRIPTION = 'Perhaps you are wondering "How do the models make decisions?" There are a lot of ways of gaining deeper \
                       understanding: sometimes models are inherently explainable, and sometimes we need advanced explanation techniques. \
                       Let\'s look at some of the models from the previous tab and try to get insight for their behaviour.'
    XAI_LOGREG_DESCRIPTION = ['Logistic regression can be explained fairly easily. Let\'s look at the decision function:',
                              FORMULA_TOKEN + r'f(\textbf{x}) = \frac{1}{1 + \exp{(- \textbf{w}^\text{T}\textbf{x} + b)}},',
                              'where *f* outputs the probabillity of the person being ill, **x** is the feature vector, \
                               and **w** is the weight vector which defines our logistic regression model. We can show that',
                               FORMULA_TOKEN + r'\nabla_\textbf{x} f(\textbf{x}) = f(\textbf{x}) (1 - f(\textbf{x})) \textbf{w} \sim  \textbf{w}.',
                               'We can see that for each given input the small individual perturbations of the features lead to \
                               changes of the decision function proportional to **w**. Since all features are normalized in preprocessing, \
                               weights can tell us about the relevance of all features for model\'s decision-making. Here is the plot:'
                             ]
    XAI_LOGREG_DISCUSSION = 'Weights with higher absolute value correspond to more relevant features; features with positive weights \
                             (blue) shift the model\'s decision towards the "ill" label, features with negative weights (red) \
                             shift the model\'s decision towards the "healthy" label.'
    XAI_BOOST_DESCRIPTION = 'Gradient Boosting is a black-box model, meaning that it doesn\'t have a weight vector or some other way \
                             of being interpretable by default. We will use the SHAP values to gain some insight. SHAP values, in essence, \
                             tell us how a prediction would change if we were to "hide" the value of a certain feature from the model. \
                             Let\'s see them in action now!'
    XAI_BOOST_INDIVIDUAL_DISCUSSION = 'On the plot you can see the features which contributed the most to the prediction result. The red features \
                                       contribute to lableling the person as ill, while the blue features contribute to labeling the person as \
                                       healthy. Among red features you can often see things like high BMI, low health self-assessment or \
                                       high blood pressure. Among blue features you can often see things like normal cholesterol levels, \
                                       healthy BMI or high health self-assessment.'
    XAI_BOOST_GLOBAL_DISCUSSION = 'Here we can see the average relevance for different values of different features. If the feature is located low, \
                                   all its values are located near the 0.0 line, which means that these features are almost irrelevant. Higher features \
                                   have higher importance. Let\'s look at a few examples.'
    XAI_BOOST_BMI_DISCUSSION = '**BMI**: We can see that the highest BMI values often have maximal relevance and shift the model to label the person \
                                as ill. At the same time, the lowest BMI values are slightly less relevant and encourage the model to label \
                                the person as healthy. Average values of BMI tend to have low relevance.'
    XAI_BOOST_STROKE_DISCUSSION = '**Stroke**: *Stroke* is a binary variable. We can see that if a person didn\'t suffer a stroke (*Stroke* = 0), \
                                   the feature is not at all relevant for the model. If the person did suffer a stroke (*Stroke* = 1), \
                                   the feature may be somewhat relevant for the prediction, and shifts the decision towards an \'ill\' label.'
    
    # Feature exploration dictionaries
    FEATURES_DESC = {'age': 'The patient\'s age in years',
                     'sex': 'Biological sex',
                     'cp': 'Chest pain type',
                     'trestbps': 'Resting blood pressure (on admission to the hospital)',
                     'chol': 'Serum cholestoral in **mg/dl**',
                     'fbs': 'Is fasting blood sugar higher than 120 **mg/dl**?',
                     'restecg': 'Resting electrocardiographic results',
                     'thalach': 'Maximum heart rate achieved',
                     'exang': 'Exercise induced angina',
                     'oldpeak': 'ST depression induced by exercise relative to rest. For more details see \
                                https://en.wikipedia.org/wiki/ST_depression',
                     'slope': 'The slope of the peak exercise ST segment. For more details see \
                               https://en.wikipedia.org/wiki/ST_segment',
                     'ca': 'Number of major vessels colored by flourosopy',
                     'thal': 'No damn clue what this is bois',
                     'num': 'Presence of heart disease'}
    FEATURES_VALS = {'age': 'Ordered integer value in range $[29, 77]$',
                     'sex': '0 = female, 1 = male',
                     'cp': '1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic. For details visit \
                            https://www.textbookofcardiology.org/wiki/Chest_Pain_/_Angina_Pectoris',
                      'trestbps': 'Ordered integer value in range $[94, 200]$',
                      'chol': 'Ordered integer value in range $[126, 564]$',
                      'fbs': '0 = no, 1 = yes',
                      'restecg': '0 = normal, 1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or \
                       depression of > 0.05 mV), 2 = showing probable or definite left ventricular hypertrophy by Estes\' criteria',
                      'thalach': 'Ordered integer value in range $[71, 202]$',
                      'exang': '0 = absent, 1 = present',
                      'oldpeak': 'Continuous value in range $[0.0, 6.2]$',
                      'slope': '1 = upsloping, 2 = flat, 3 = downsloping',
                      'ca': 'Amount of vessels (0-3)',
                      'thal': '3 = normal, 6 = fixed defect, 7 = reversable defect',
                      'num': '0 - healthy, 1-4 - ill'}
    
    # Error messages
    EMPTY_FILTER_ERROR = 'Please select a non-empty filtered set 😭😭😭'
    MISSING_FEATURE_ERROR = '😨😳😖 Whoops! Unexpected key error - please contact the developer'
    NO_LOGREG_ERROR = '🤷‍♂️🤷‍♀️ No trained logistic regression detected - please visit the previous tab to train the model!'
    NO_BOOST_ERROR = '🤷‍♂️🤷‍♀️ No trained gradient boosting detected - please visit the previous tab to train the model!'
    TRIVIAL_BOOST_ERROR = '❗ **WARNING** ❗ A trivial classifier was learned - evaluation impossible.'
    OVERFIT_WARNING = '🕵️‍♂️ *Automated Overfit Detection*: final losses are disparate for train and test data. \
                        Using a simpler model is advised.'

    # Miscellaneous
    SEPARATOR = '_' * 15