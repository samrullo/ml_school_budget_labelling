import os
import pandas as pd
import numpy as np
from utils.train_test_split import multilabel_train_test_split

train_df = pd.read_csv(os.path.relpath("datasets/school_budget_labelling_train_dataset.csv"))

# view data types of the columns
print('lets explore datatypes of the dataframe columns')
print(train_df.dtypes.value_counts())

LABELS = ['Function',
          'Use',
          'Sharing',
          'Reporting',
          'Student_Type',
          'Position_Type',
          'Object_Type',
          'Pre_K',
          'Operating_Status']

feature_labels = ['Unnamed: 0', 'Object_Description', 'Text_2', 'SubFund_Description',
                  'Job_Title_Description', 'Text_3', 'Text_4', 'Sub_Object_Description',
                  'Location_Description', 'FTE', 'Function_Description',
                  'Facility_or_Department', 'Position_Extra', 'Total',
                  'Program_Description', 'Fund_Description', 'Text_1']

# count how many unique values each LABEL has
unique_label_count_df = train_df[LABELS].apply(pd.Series.nunique)

# print out unique label count dataframe
print("Number of categories for each label. Labels is a list of labels that are used to label each budget item.")
print(unique_label_count_df)

# convert label datatypes to categorical
train_df['Function'] = train_df['Function'].astype('category')
print("After converting Function column data type to category. Lets see what its datatype looks like")
print(train_df['Function'].dtype)

# let's convert all labels' datatypes to categorical
train_df[LABELS] = train_df[LABELS].apply(lambda x: x.astype('category'))
print('After converting all LABELs datatypes to categorical lets see their datatypes')
print(train_df[LABELS].dtypes)

# now let's create label_dummies by converting categorical labels to dummies
label_dummies_df = pd.get_dummies(train_df[LABELS], prefix_sep='_')
print("label_dummies shape:{}".format(label_dummies_df.values.shape))

NUMERIC_COLS = ['FTE', 'Total']

X_train, X_test, Y_train, Y_test = multilabel_train_test_split(train_df[NUMERIC_COLS], label_dummies_df, size=1000, min_count=10, seed=123)
