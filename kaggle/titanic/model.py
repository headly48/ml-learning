import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf


# Load dataset.
dftrain = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/train.csv'))
dfeval = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/test.csv'))

dftrain.Cabin = dftrain.Cabin.fillna('')
dfeval.Cabin = dfeval.Cabin.fillna('')

dftrain.Embarked = dftrain.Embarked.fillna('')
dfeval.Embarked = dfeval.Embarked.fillna('')
# dftrain.dropna(how="any",inplace = True)
# dfeval.dropna(how="any",inplace = True)

dftrain.pop('PassengerId')

print(dfeval)

dfeval['Survived'] = 1
trainingPassengerIds = dfeval.pop('PassengerId')


survived_test = dfeval.pop('Survived')

# dftrain.dropna(how="any",inplace = True)
# dfeval.dropna(how="any",inplace = True)

y_train = dftrain.pop('Survived')

# y_eval = dfeval.pop('survived')

# Create the pandas DataFrame
# survived_test = pd.Series(data=([0] * (trainingPassengerIds.shape[0] + 1)), name='Survived', dtype='int64')

print(survived_test)
# print(survived_test.dtypes)
# print(survived_test.__len__)

# print(y_train)
# print(y_train.dtypes)
# print(y_train.__len__)

# print(y_train.)

# RE ADD 'Cabin'
CATEGORICAL_COLUMNS = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
NUMERIC_COLUMNS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)

eval_input_fn = make_input_fn(dfeval, survived_test, num_epochs=1, shuffle=False)

age_x_gender = tf.feature_column.crossed_column(['Age', 'Sex'], hash_bucket_size=256)

derived_feature_columns = [age_x_gender]

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)

# result = linear_est.evaluate(eval_input_fn)

# print(y_train)

# results = list(linear_est.predict(eval_input_fn))


# print(results[0])

# print(len(results))
# for result, index in enumerate(results):
#     print(result)

# for result in results:
#   print(result['probabilities'])


# print(result[0])