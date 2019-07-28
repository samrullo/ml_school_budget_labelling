import os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from utils.train_test_split import multilabel_train_test_split
from utils.utils import combine_text_data
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


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

text_df = combine_text_data(df, to_drop=NUMERIC_COLS + LABELS + ['Unnamed: 0'])

TOKEN_BASIC = '\\S+(?=\\S+)'
vec_whitespace = CountVectorizer(token_pattern=TOKEN_BASIC)

TOKEN_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\S+)'
vec_alphanumeric = CountVectorizer(token_pattern=TOKEN_ALPHANUMERIC)

vec_whitespace.fit_transform(text_df)
logging.info("There are {} tokens when we tokenize with basic token pattern".format(len(vec_whitespace.get_feature_names())))

vec_alphanumeric.fit_transform(text_df)
logging.info("There are {} tokens when we tokenize with alphanumeric token pattern".format(len(vec_alphanumeric.get_feature_names())))