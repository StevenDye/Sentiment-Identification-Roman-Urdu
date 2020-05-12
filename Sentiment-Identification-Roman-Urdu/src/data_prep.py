"""This module is for data cleaning"""

import re
import numpy as np
import pandas as pd
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.stopwords import STOPWORDS


def create_meta_features(df):
    """This function creates meta features for EDA analysis"""

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

    return df


# Data is from http://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set#
df = pd.read_csv('data/Roman Urdu DataSet.csv',
                 names=["text", "sentiment", "third"])  # Read datafile


# Final column does not appear to to provide any useful information. We can drop it
df = df.drop(columns=['third'])  # Drop empty column
df = df.dropna()  # Drop empty rows
df['sentiment'] = df['sentiment'].replace('Neative', 'Negative')  # Fix typo
df = df.drop_duplicates()

# Set target and features
X = df.drop(columns=['sentiment'])
y = df['sentiment']

enocder = LabelEncoder()
y = enocder.fit_transform(y)
# 2 postive 1 nuetral 0 negative

categories = ['Positive', 'Neutral', 'Negative']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

train_df = pd.merge(X_train.reset_index(drop=True),
                    pd.DataFrame(y_train, columns=['sentiment']),
                    left_index=True, right_index=True)

test_df = pd.merge(X_test.reset_index(drop=True),
                   pd.DataFrame(y_test, columns=['sentiment']),
                   left_index=True, right_index=True)

# Make dataframes for EDA
EDA_df = create_meta_features(df)
EDA_df_train = create_meta_features(train_df)
EDA_df_test = create_meta_features(test_df)

# Need to Corpus of words
corpus = []
for i in range(0, len(X_train)):
    review = re.sub('[^a-zA-Z]', ' ', X_train.iloc[:, 0].values[i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    corpus.append(review)

# tokenize corpus
tokenized = []
for i in range(0, len(corpus)):
    tokenized.append(word_tokenize(corpus[i]))

articles_concat = []
for article in tokenized:
    articles_concat += article

articles_freqdist = FreqDist(articles_concat)
art_freqdist_50 = articles_freqdist.most_common(50)

# vectorize training and test data
vectorizer = TfidfVectorizer()
# vectorizer_20000 = TfidfVectorizer(max_features=20000)

tf_idf_data_train = vectorizer.fit_transform(corpus)
tf_idf_data_test = vectorizer.transform(X_test['text'])