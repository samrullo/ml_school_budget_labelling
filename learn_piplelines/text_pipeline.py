import os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from utils.train_test_split import multilabel_train_test_split
from utils.utils import combine_text_data
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

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

df[LABELS] = df[LABELS].apply(lambda x: x.astype('category'))
logging.info("Converted LABELS datatypes to categorical")

NUMERIC_COLS = ['FTE', 'Total']

text_df = combine_text_data(df, to_drop=NUMERIC_COLS + LABELS + ['Unnamed: 0'])

X_train, X_test, y_train, y_test = multilabel_train_test_split(text_df, pd.get_dummies(df[LABELS]), size=0.2, min_count=1, seed=123)

pl = Pipeline([('vec', CountVectorizer()), ('clf', OneVsRestClassifier(LogisticRegression()))])

pl.fit(X_train, y_train)

logging.info("Model score : {}".format(pl.score(X_test, y_test)))
