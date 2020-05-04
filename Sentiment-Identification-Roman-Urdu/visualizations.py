"""This module holds functions that plot visualizations"""

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def meta_feature_distribution(train_df, test_df):
    """
    This function plots the distributions of the meta feature for each
    sentiment value in the training data and for the totals in the
    training and testing data
    """

    META_FEATURES = ['word_count', 'unique_word_count', 'stop_word_count',
                     'mean_word_length', 'char_count']

    POSITIVE = train_df['sentiment'] == 2
    NEUTRAL = train_df['sentiment'] == 1
    NEGATIVE = train_df['sentiment'] == 0

    fig, axes = plt.subplots(ncols=2, nrows=len(META_FEATURES),
                             figsize=(20, 50), dpi=100)

    for i, feature in enumerate(META_FEATURES):
        sns.distplot(train_df.loc[POSITIVE][feature],
                     label='Positive', ax=axes[i][0], color='blue')
        sns.distplot(train_df.loc[NEUTRAL][feature],
                     label='Neutral', ax=axes[i][0], color='gray')
        sns.distplot(train_df.loc[NEGATIVE][feature],
                     label='Negative', ax=axes[i][0], color='red')

        sns.distplot(train_df[feature], label='Training', ax=axes[i][1])
        sns.distplot(test_df[feature], label='Test', ax=axes[i][1])

        for j in range(2):
            axes[i][j].set_xlabel('')
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].legend()

        axes[i][0].set_title(f'{feature} Target Distribution in Training Set',
                             fontsize=13)
        axes[i][1].set_title(f'{feature} Training & Test Set Distribution',
                             fontsize=13)

    plt.show()
    

def evaluate_model(X_train, y_train, X_test, y_test, model):
    """This function displays model scores"""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    diff = abs(scores.mean() - model.score(X_test, y_test))
    SD = diff / scores.std()
    
    print(f"Training Score:{model.score(X_train, y_train)}")
    print(f"Cross V Score: {scores.mean()} +/- {scores.std()}")
    print(f"Testing Score: {model.score(X_test, y_test)}")
    print(f"Cross & Test Diff: {diff}")
    print(f"Standard Deviations Away: {SD}")
    print(confusion_matrix(y_test, preds))


def confusion_matrix_heat_map(clf, X_test, y_test):
    """
    Function takes a classifier clf as an argument to create a
    confusion matrix.  The function then normalizes the data across
    the confusion matrix and corrects a known bug in matplotlib that
    incorrectly cuts off the top and bottom rows of the heat map.
    Parameters:
    clf: Classifier already fit to training data
    X_test: Features for test data
    y_test: Predicted outputs for X_test
    """
    # create confusion matrix <cm>
    cm = metrics.confusion_matrix(clf.predict(X_test), y_test)
    # create normalized confusion matrix <cm_nor>
    cm_nor = np.zeros((cm.shape[0], cm.shape[1]))
    for col in range(cm.shape[1]):
        cm_nor[:, col] = (cm[:, col] / sum(cm[:, col]))
    plt.ylim(-10, 10)
    # create normalized confusion matrix heat map
    sns.heatmap(cm_nor, cmap="Blues", annot=True, annot_kws={"size": 8})
    locs, labels = plt.xticks()
    plt.xticks(locs, ("Pos", "Neu", "Neg"))
    locs, labels = plt.yticks()
    plt.yticks(locs, ("Pos", "Neu", "Neg"))
    plt.yticks(rotation=0)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Sentiment Classification Percentage")
    # known bug in matplotlib chops off a portion of the
    # top and bottom rows of heat maps.  This section of
    # code recovers the top and bottom limits and moves them
    # so that the map displays appropriately.
    bottom, top = plt.ylim()
    bottom += 0.5
    top -= 0.5
    plt.ylim(bottom, top)
    plt.show()
