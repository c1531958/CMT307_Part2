# Some parts of code taken from Session 2 and 3 from module CMT307 Applied Machine Learning

import os

# must install
import pandas as pd
import nltk


from sklearn.feature_extraction.text import CountVectorizer
import nltk.sentiment.sentiment_analyzer
import numpy as np

# for feature selection
from sklearn import svm

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataPreprocessing import getVocabulary, getXY, selectBestFeatures
from featureSelection import featureSelection, tryClassifiers

# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('universal_tagset')

def main():
    # linux
    path_train_neg = os.getcwd()+'/datasets_coursework1/IMDb/train/imdb_train_neg.txt'
    path_train_pos = os.getcwd()+'/datasets_coursework1/IMDb/train/imdb_train_pos.txt'

    path_dev_neg = os.getcwd()+'/datasets_coursework1/IMDb/dev/imdb_dev_neg.txt'
    path_dev_pos = os.getcwd()+'/datasets_coursework1/IMDb/dev/imdb_dev_pos.txt'

    path_test_neg = os.getcwd()+'/datasets_coursework1/IMDb/test/imdb_test_neg.txt'
    path_test_pos = os.getcwd()+'/datasets_coursework1/IMDb/test/imdb_test_pos.txt'

    # windows
    #path_dev_neg = os.getcwd()+'\\datasets_coursework1\\IMDb\\dev\\imdb_dev_neg.txt'
    #path_dev_pos = os.getcwd()+'\\datasets_coursework1/IMDb/dev/imdb_dev_pos.txt'

    #path_test_neg = os.getcwd()+'\\datasets_coursework1\\IMDb\\test\\imdb_test_neg.txt'
    #path_test_pos = os.getcwd()+'\\datasets_coursework1\\IMDb\\test\\imdb_test_pos.txt'

    #path_train_neg = os.getcwd()+'\\datasets_coursework1\\IMDb\\train\\imdb_train_neg.txt'
    #path_train_pos = os.getcwd()+'\\datasets_coursework1\\IMDb\\train\\imdb_train_pos.txt'


    df_train_pos = open(path_train_pos, encoding="utf8").read().splitlines()
    df_train_neg = open(path_train_neg, encoding="utf8").read().splitlines()

    df_dev_pos = open(path_dev_pos, encoding="utf8").read().splitlines()
    df_dev_neg = open(path_dev_neg, encoding="utf8").read().splitlines()

    df_test_pos = open(path_test_pos, encoding="utf8").read().splitlines()
    df_test_neg = open(path_test_neg, encoding="utf8").read().splitlines()


    clf_svm = svm.SVC(C=1, gamma='auto', random_state=42)
    training_set = df_train_pos + df_train_neg


    # Feature selection
    list_num_features = [250, 500, 750, 1000]
    best_vocabulary, best_X_train, best_Y_train = featureSelection( list_num_features, training_set,
                                                                    df_train_pos, df_train_neg,
                                                                    clf_svm,
                                                                    df_dev_pos, df_dev_neg)#,500,750, 1000]



    # Try different types of classifiers
    best_clf = tryClassifiers( df_dev_pos, df_dev_neg,
                               best_vocabulary, clf_svm,
                               best_X_train, best_Y_train)

    X, Y = getXY(df_test_pos, df_test_neg, best_vocabulary)
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(X)


    stats_best = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        best_clf.fit(X_train, Y_train)
        Y_pred = best_clf.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test, Y_pred, average='macro')

        stats = [accuracy, precision, recall, fscore]
        stats_best.append(stats)


    folds_index = ['k=' + str (i) for i in range(1, folds+1)]
    cols = ['accuracy', 'precision', 'recall', 'fscore']
    df_best = pd.DataFrame(stats_best, columns=cols, index=folds_index)

    folds_index.append('mean')
    mean_df = [ df_best['accuracy'].mean(),
                        df_best['precision'].mean(),
                        df_best['recall'].mean(),
                        df_best['fscore'].mean()
                    ]

    print(pd.DataFrame([mean_df], columns=cols, index=['mean']))

main()
