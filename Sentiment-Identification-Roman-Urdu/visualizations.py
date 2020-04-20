"""This module holds functions that plot visualizations"""

import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import train_df, test_df

def meta_feature_distribution():
    """This function plots the distributions of the meta feature for each sentiment value in the training data and for the totals in the training and testing data"""
    
    METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count',
                'mean_word_length', 'char_count']

    POSITIVE = train_df['sentiment'] == 2
    NEUTRAL = train_df['sentiment'] == 1
    NEGATIVE = train_df['sentiment'] == 0


    fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

    for i, feature in enumerate(METAFEATURES):
        sns.distplot(train_df.loc[POSITIVE][feature], label='Positive', ax=axes[i][0], color='blue')
        sns.distplot(train_df.loc[NEUTRAL][feature], label='Neutral', ax=axes[i][0], color='gray')
        sns.distplot(train_df.loc[NEGATIVE][feature], label='Negative', ax=axes[i][0], color='red')

        sns.distplot(train_df[feature], label='Training', ax=axes[i][1])
        sns.distplot(test_df[feature], label='Test', ax=axes[i][1])
    
        for j in range(2):
            axes[i][j].set_xlabel('')
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].legend()
    
        axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
        axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

    plt.show()