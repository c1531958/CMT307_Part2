from dataPreprocessing import getVocabulary, getXY, selectBestFeatures
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

cols = ['accuracy', 'precision', 'recall', 'fscore']

def featureSelection(list_num_features, training_set, df_train_pos, df_train_neg, clf_svm, df_dev_pos, df_dev_neg):
    best_fscore_dev = 0.0
    best_vocabulary = None
    best_X_train = None
    best_Y_train = None
    vocabulary = getVocabulary(training_set)
    X_train, Y_train = getXY(df_train_pos, df_train_neg, vocabulary)
    stats_dev = []

    for num_features in list_num_features:
        print('Testing num_features = ' + str(num_features))

        # Select best k features
        k_feature_X_train, features_selected = selectBestFeatures(X_train, Y_train, num_features)
        Y_train_array=np.asarray(Y_train)

        clf_svm.fit(k_feature_X_train, Y_train_array)
        
        # retrieve indexes from bool array 
        indexes_selected = np.where(features_selected)[0]
        # vocabulary fork k features
        k_feature_vocabulary = [vocabulary[i] for i in indexes_selected]
        X_dev, Y_dev  = getXY(df_dev_pos, df_dev_neg, k_feature_vocabulary)
        
        # Transform to vector
        # X_dev=np.asarray(X_dev)
        Y_dev_predictions = clf_svm.predict(X_dev)

        # Calculate performance measures
        accuracy_dev = accuracy_score(Y_dev, Y_dev_predictions)
        precision_dev, recall_dev, fscore_dev, support_dev = precision_recall_fscore_support(Y_dev, Y_dev_predictions, average='macro')
        precision_dev_rf, recall_dev_rf, fscore_dev_rf, support_dev_rf = precision_recall_fscore_support(Y_dev,
         Y_dev_predictions, average='macro')

        stats = [accuracy_dev, precision_dev, recall_dev, fscore_dev]
        stats_dev.append(stats)

        # update best fscore
        if fscore_dev>=best_fscore_dev:
            best_fscore_dev=fscore_dev
            best_num_features=num_features
            best_vocabulary=k_feature_vocabulary
            best_X_train = k_feature_X_train
            best_Y_train = Y_train_array


    print('Best f-score is for num_features: ' + str(best_num_features))
    df = pd.DataFrame(stats_dev, columns=cols, index=list_num_features)
    print(df)

    return best_vocabulary, best_X_train, best_Y_train

def tryClassifiers(df_dev_pos, df_dev_neg, best_vocabulary, clf_svm, best_X_train, best_Y_train):
    clf_rf = RandomForestClassifier(n_estimators=100, 
                                    max_depth=2,
                                    random_state=0)

    clf_lr = LogisticRegression(random_state=0,
                                solver='lbfgs',
                                max_iter=1000)

    stats_classifiers = []
    X_dev, Y_dev  = getXY(df_dev_pos, df_dev_neg, best_vocabulary)

    classifiers = [clf_svm, clf_rf, clf_lr]
    best_fscore_clf = 0.0
    best_clf = None
    for classifier in classifiers:
        classifier.fit(best_X_train, best_Y_train)

        Y_pred = classifier.predict(X_dev)

        accuracy_dev = accuracy_score(Y_dev, Y_pred)
        precision_dev, recall_dev, fscore_dev, support_dev = precision_recall_fscore_support(Y_dev, Y_pred, average='macro')

        stats = [accuracy_dev, precision_dev, recall_dev, fscore_dev]
        stats_classifiers.append(stats)

        # update best fscore
        if fscore_dev >= best_fscore_clf:
            best_fscore_clf = fscore_dev
            best_clf = classifier

    print('Best f-score is for classifier: ' + str(type(classifier)))
    df_classifiers = pd.DataFrame(stats_classifiers, columns=cols, index=['svm', 'random_forest', 'logistic_regression'])
    print(df_classifiers)
    return classifier