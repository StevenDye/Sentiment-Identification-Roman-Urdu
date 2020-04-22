"""This module is for data cleaning"""

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from stopwords import STOPWORDS

# Data is from http://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set#
df = pd.read_csv('data/Roman Urdu DataSet.csv',
                 names=["text", "sentiment", "third"])  # Read datafile


# Final column does not appear to to provide any useful information. We can drop it
df = df.drop(columns=['third'])  # Drop empty column
df = df.dropna()  # Drop empty rows
df['sentiment'] = df['sentiment'].replace('Neative', 'Negative')  # Fix typo

# lowercase all text
df['text'] = df.apply(lambda row: row['text'].lower(), axis=1)

# Feature Engineering
# word count
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
# unique_word_count
df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))
# stop_word_count
df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
# mean_word_length
df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# char_count
df['char_count'] = df['text'].apply(lambda x: len(str(x)))
# Some text are just the space character, which gives NaN values for mean_word_length
df = df.fillna(0)  # Fill NaNs

# Set target and features
X = df.drop(columns=['sentiment'])
y = df['sentiment']

enocder = LabelEncoder()
y = enocder.fit_transform(y)
# 2 postive 1 nuetral 0 negative

# Spliting dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Make dataframes for EDA
train_df = pd.merge(X_train.reset_index(drop=True),
                    pd.DataFrame(y_train, columns=['sentiment']),
                    left_index=True, right_index=True)

test_df = pd.merge(X_test.reset_index(drop=True),
                   pd.DataFrame(y_test, columns=['sentiment']),
                   left_index=True, right_index=True)


# Vectorize words
count_vectorizer = CountVectorizer()
# tfidf_vectorizer = TfidfVectorizer()
# Will need to determine which one of these will be better
train_vectors = count_vectorizer.fit_transform(X_train["text"])

# We're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors
test_vectors = count_vectorizer.transform(X_test["text"])

# Combined vectorized words with meta features
X_train_combined = np.concatenate((X_train.drop(columns=['text']),
                                   train_vectors.toarray()), axis=1)
X_test_combined = np.concatenate((X_test.drop(columns=['text']),
                                  test_vectors.toarray()), axis=1)

# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(X_train_combined, y_train)

# Takes about 30 minutes
#smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_resample(X_train_combined, y_train)


# Scale data
scaler = StandardScaler()  # with_mean=False Sparse matrixices need with_mean=False
X_train_scale = scaler.fit_transform(X_train_combined)
X_test_scale = scaler.transform(X_test_combined)

# Create sythetic data to balance outputs
# WARNING: takes about 25 minutes on your personal computer to create the synthetic data
smt = SMOTE()
X_train_smote, y_train = smt.fit_sample(X_train_scale, y_train)
