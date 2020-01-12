# Some parts of code taken from Session 2 and 3 from module CMT307 Applied Machine Learning

import os
import sys

import nltk
import numpy as np
from sklearn import svm

from dataPreprocessing import getVocabulary, getXY, selectBestFeatures
from featureSelection import featureSelection, tryClassifiers
from testPerformance import kFoldCrossValidation

# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('universal_tagset')

def main(folder_name):
    # linux
    path_train_neg = os.getcwd()+'/'+folder_name+'/train/imdb_train_neg.txt'
    path_train_pos = os.getcwd()+'/'+folder_name+'/train/imdb_train_pos.txt'

    path_dev_neg = os.getcwd()+'/'+folder_name+'/dev/imdb_dev_neg.txt'
    path_dev_pos = os.getcwd()+'/'+folder_name+'/dev/imdb_dev_pos.txt'

    path_test_neg = os.getcwd()+'/'+folder_name+'/test/imdb_test_neg.txt'
    path_test_pos = os.getcwd()+'/'+folder_name+'/test/imdb_test_pos.txt'

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

    # k-fold cross validation for test data
    folds = 5
    test(folds, df_test_pos, df_test_neg)

if __name__ == "__main__":
    main(sys.argv[1])
