from dataPreprocessing import getXY
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
import pandas as pd

def kFoldCrossValidation(folds, df_test_pos, df_test_neg):
    X, Y = getXY(df_test_pos, df_test_neg, best_vocabulary)
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(X)


    # k-fold cross validation on test set
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

    # find mean accuracy, precision, recall and f-score
    folds_index.append('mean')
    mean_df = [ df_best['accuracy'].mean(),
                df_best['precision'].mean(),
                df_best['recall'].mean(),
                df_best['fscore'].mean()
            ]

    print(pd.DataFrame([mean_df], columns=cols, index=['mean']))