import folktables
from folktables import ACSDataSource
import numpy as np

from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing

import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn

# TODO: have it load data from file when False
GET_DATA = False

SENSITIVE_ATTRIBUTE = 'SEX'


#(Age) must be greater than 16 and less than 90, and (Person weight) must be greater than or equal to 1
def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]
    df = df[df['PWGTP'] >= 1]
    return df

ACSEmployment = folktables.BasicProblem(
    features=[
    'AGEP', #age; for range of values of features please check Appendix B.4 of 
            #Retiring Adult: New Datasets for Fair Machine Learning NeurIPS 2021 paper
    'SCHL', #educational attainment
    'MAR', #marital status
    'RELP', #relationship
    'DIS', #disability recode
    'ESP', #employment status of parents
    'CIT', #citizenship status
    'MIG', #mobility status (lived here 1 year ago)
    'MIL', #military service
    'ANC', #ancestry recode
    'NATIVITY', #nativity
    'DEAR', #hearing difficulty
    'DEYE', #vision difficulty
    'DREM', #cognitive difficulty
    'SEX', #sex
    'RAC1P', #recoded detailed race code
    'GCL', #grandparents living with grandchildren
    ],
    target='ESR', #employment status recode
    target_transform=lambda x: x == 1,
    group=SENSITIVE_ATTRIBUTE, 
    preprocess=employment_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

if GET_DATA:
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True) #data for California state
else:
    data = pd.read_csv('data/2018/1-Year/psam_p06.csv')
    acs_data = data


features, label, group = ACSEmployment.df_to_numpy(acs_data)
data = pd.DataFrame(features, columns = ACSEmployment.features)
data['label'] = label



# Once the data is loaded, we need to partition it into train, validation, and test sets.
# We want 70/30 train/test split, and 20% of the training set to be used for validation.
# We want to randomly split the training set 5 times.
# We also want this data to be in the format that AIF360 expects, which should also handle preprocessing.

train, test = train_test_split(data, test_size=0.3, random_state=42)
train_val_splits = dict()
favorable_classes = [True]
protected_attribute_names = [ACSEmployment.group]
privileged_classes = np.array([[1]])
privileged_groups = [{'SEX': 1}]
unprivileged_groups = [{'SEX': 2}]

test = StandardDataset(test, 'label', favorable_classes = favorable_classes,
                        protected_attribute_names = protected_attribute_names,
                        privileged_classes = privileged_classes)

for i in range(5):
    train_split, val_split = train_test_split(train, test_size=0.2, random_state=42)
    train_split = StandardDataset(train_split, 'label', favorable_classes = favorable_classes,
                        protected_attribute_names = protected_attribute_names,
                        privileged_classes = privileged_classes)
    val_split = StandardDataset(val_split, 'label', favorable_classes = favorable_classes,
                        protected_attribute_names = protected_attribute_names,
                        privileged_classes = privileged_classes)
    train_val_splits[i] = (train_split, val_split)


# Next, train a model on each of the 5 train/val splits.
# We will use a simple logistic regression model. (Unsure if this is complex enough for the task?)
# We will perform hyperparameter tuning, averaging the results of the 5 train/val splits.
# We will evaluate each model twice: once using accuracy, and again using the AIF360 metric.
# We will save the best model based on accuracy, and the best model based on the AIF360 metric.

hyperparameters = { 'C': [0.01, 0.1, 1, 10, 100]}

# Perform a grid search over the hyperparameters (Can change this later)
for C in hyperparameters['C']:
    # Train a model on each of the 5 train/val splits
    for i in range(5):
        train_train = train_val_splits[i][0]
        train_val = train_val_splits[i][1]
        model = sklearn.linear_model.LogisticRegression(C=C, solver='liblinear')
        model.fit(train_train.features, train_val.labels.ravel())
        # Evaluate the model on the validation set
        val_predictions = train_val.copy()
        val_predictions.labels = model.predict(train_val.features)
        val_accuracy = sklearn.metrics.accuracy_score(train_val.labels, val_predictions)
        metric = ClassificationMetric(train_val, val_predictions, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        metric_arrs = {}
        #Statistical Parity Difference measures the difference of the above values instead of ratios, hence we
        #would like it to be close to 0.
        metric_arrs['stat_par_diff']=(metric.statistical_parity_difference())

    print("C: ", C)
    print("Validation Accuracy: ", val_accuracy)
    print("Validation Statistical Parity Difference: ", metric_arrs['stat_par_diff'])