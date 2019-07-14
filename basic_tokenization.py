import os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from utils.train_test_split import multilabel_train_test_split
import logging

_logger = logging.basicConfig(level=logging.INFO)

# let's load data...
df = pd.read_csv(os.path.relpath("datasets/school_budget_labelling_train_dataset.csv"))
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

df.Position_Extra.fillna('', inplace=True)

TOKEN_BASIC = '\\S+(?=\\S+)'
vec_whitespace = CountVectorizer(token_pattern=TOKEN_BASIC)
vec_whitespace.fit(df['Position_Extra'])
msg="There are {} tokens in Position_Extra when tokenizing on whitespace"
print(msg.format(len(vec_whitespace.get_feature_names())))
print("First 15 tokens:{}".format(vec_whitespace.get_feature_names()[:15]))

TOKEN_ALPHANUMERIC='[A-Za-z0-9]+(?=\\S+)'
vec_alphanumeric=CountVectorizer(token_pattern=TOKEN_ALPHANUMERIC)
vec_alphanumeric.fit(df['Position_Extra'])
msg="There are {} tokens in Position_Extra when tokenizing on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print("First 15 token:{}".format(vec_alphanumeric.get_feature_names()))