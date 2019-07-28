import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
from utils.train_test_split import multilabel_train_test_split
import logging

_logger = logging.basicConfig(level=logging.INFO)

# let's load data...
df = pd.read_csv(os.path.relpath("../datasets/school_budget_labelling_train_dataset.csv"))
df = df.loc[np.random.choice(df.index, 4500)]
logging.info("Finished loading data")

LABELS = ['Function',
          'Use',
          'Sharing',
          'Reporting',
          'Student_Type',
          'Position_Type',
          'Object_Type',
          'Pre_K',
          'Operating_Status']

NUMERIC_COLS = ['FTE', 'Total']

numeric_only_data = df[NUMERIC_COLS].fillna(-1000)
# convert LABELS into categorical datatype
df[LABELS] = df[LABELS].apply(lambda x: x.astype('category'))
label_dummies = pd.get_dummies(df[LABELS])

# split data into train and test set
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_only_data, label_dummies, size=0.2, min_count=1, seed=123)
logging.info("Finished splitting data to train, test set")

# create one step pipline that consists of just one step which is OneVsRestClassifier
# a step of a pipeline is a tuple with two elements. First element is the name of the step. Second element is an object with fit and transform methods

pl = Pipeline([('clf', OneVsRestClassifier(LogisticRegression()))])

logging.info("Begin fitting model")
pl.fit(X_train, y_train)

print("Our score:{}".format(pl.score(X_test, y_test)))
