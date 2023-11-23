from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import data_loader_utility as data_load
import matplotlib.pyplot as plt

# Need to add data loading and finish implementing initial lda model so I can build models
train_load, val_load, test_load = data_load.get_dataloaders()
X_train, y_train, X_test, y_test = data_load.get_tabular_data()
X_train_pd = pd.DataFrame(X_train)
X_test_pd = pd.DataFrame(X_test)
y_train_arr = np.array(y_train)
y_test_arr = np.array(y_test)


def lda(features, targets, test_features, n_components):
    """Train LDA on training data with specified number of components. Returns training and test sets both transformed
    by the LDA model built on the training set, as well as the model used."""
    np.random.seed(64)

    # transform train data
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    new_train_features = lda.fit_transform(features, np.array(targets))

    # transform test data
    new_test_features = lda.transform(test_features)

    return new_train_features, new_test_features, lda


def k_nearest_neighbors(xtrain, xtest, ytrain, ytest, grid_to_search={}):
    """Fit a KNN classifier and display performance metrics (acc/prec/rec/f1) and confusion matrix"""
    np.random.seed(64)

    knn = KNeighborsClassifier()
    try:
        optimal = GridSearchCV(knn, grid_to_search, refit=True)
        optimal.fit(xtrain, ytrain)
        internal_preds = optimal.predict(xtest)
    except InvalidParameterError as e:
        print(f'{e}\nAttempting to fit with a default KNN classifier.')
        optimal = KNeighborsClassifier()
        optimal.fit(xtrain, ytrain)
        internal_preds = optimal.predict(xtest)

    # optimal.fit(xtrain, ytrain)
    # internal_preds = optimal.predict(xtest)

    # check performance
    print('MODEL PERFORMANCE:\n-------------\naccuracy: ', metrics.accuracy_score(ytest, internal_preds),
          '\nprecision: ', metrics.precision_score(ytest, internal_preds,
                                                   labels=[int(x) for x in list(set(ytrain))], average='macro'),
          '\nrecall: ', metrics.recall_score(ytest, internal_preds,
                                             labels=[int(x) for x in list(set(ytrain))], average='macro'),
          '\nf1-score: ', metrics.f1_score(ytest, internal_preds,
                                           labels=[int(x) for x in list(set(ytrain))], average='macro'))
    confmat_internal = metrics.confusion_matrix(ytest, internal_preds)
    show_confmat_internal = metrics.ConfusionMatrixDisplay(confmat_internal)
    show_confmat_internal.plot()
    plt.show()
    return optimal


def support_vector_machine(xtrain, xtest, ytrain, ytest, grid_to_search={}):
    """Fit a support vector machine classifier and display performance metrics (acc/prec/rec/f1) and confusion matrix"""
    np.random.seed(64)

    svm = SVC()
    try:
        optimal = GridSearchCV(svm, grid_to_search, refit=True)
        optimal.fit(xtrain, ytrain)
        internal_preds = optimal.predict(xtest)
    except InvalidParameterError as e:
        print(f'{e}\nAttempting to fit with a default SVM classifier.')
        optimal = SVC()
        optimal.fit(xtrain, ytrain)
        internal_preds = optimal.predict(xtest)

    # optimal.fit(xtrain, ytrain)
    # internal_preds = optimal.predict(xtest)

    # check performance
    print('MODEL PERFORMANCE:\n-------------\naccuracy: ', metrics.accuracy_score(ytest, internal_preds),
          '\nprecision: ', metrics.precision_score(ytest, internal_preds, labels=[int(x) for x in list(set(y_train))],
                                                   average='macro'),
          '\nrecall: ',
          metrics.recall_score(ytest, internal_preds, labels=[int(x) for x in list(set(y_train))], average='macro'),
          '\nf1-score: ',
          metrics.f1_score(ytest, internal_preds, labels=[int(x) for x in list(set(y_train))], average='macro'))
    confmat_internal = metrics.confusion_matrix(ytest, internal_preds)
    show_confmat_internal = metrics.ConfusionMatrixDisplay(confmat_internal)
    show_confmat_internal.plot()
    plt.show()
    return optimal


def random_forest(xtrain, xtest, ytrain, ytest, grid_to_search={}):
    """Fit a random forest classifier and display performance metrics (acc/prec/rec/f1) and confusion matrix."""
    np.random.seed(64)

    rfc = RandomForestClassifier()
    try:
        optimal = GridSearchCV(rfc, grid_to_search, refit=True)
        optimal.fit(xtrain, ytrain)
        internal_preds = optimal.predict(xtest)
    except InvalidParameterError as e:
        print(f'{e}\nAttempting to fit with a default Random Forest classifier.')
        optimal = RandomForestClassifier()
        optimal.fit(xtrain, ytrain)
        internal_preds = optimal.predict(xtest)

    # optimal.fit(xtrain, ytrain)
    # internal_preds = optimal.predict(xtest)

    # check performance
    print('MODEL PERFORMANCE:\n-------------\naccuracy: ', metrics.accuracy_score(ytest, internal_preds),
          '\nprecision: ', metrics.precision_score(ytest, internal_preds, labels=[int(x) for x in list(set(ytrain))], average='macro'),
          '\nrecall: ', metrics.recall_score(ytest, internal_preds, labels=[int(x) for x in list(set(ytrain))], average='macro'),
          '\nf1-score: ', metrics.f1_score(ytest, internal_preds, labels=[int(x) for x in list(set(ytrain))], average='macro'))
    confmat_internal = metrics.confusion_matrix(ytest, internal_preds)
    show_confmat_internal = metrics.ConfusionMatrixDisplay(confmat_internal)
    show_confmat_internal.plot()
    plt.show()

    return optimal


LDA_X, LDA_X_test, LDA = lda(X_train, y_train, X_test,9)



# Narrow down potential k values
param_grid = {'n_neighbors': list(range(1,243,6))}
myKnn = k_nearest_neighbors(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(myKnn.best_estimator_)


# Find best k near previous result
param_grid = {'n_neighbors': list(range(25,51,2))}
myKnn = k_nearest_neighbors(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(myKnn.best_estimator_)

param_grid = {'n_neighbors': 35}
myKnn = k_nearest_neighbors(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(myKnn.best_estimator_)

# MODEL PERFORMANCE:
# -------------
# accuracy:  0.8323
# precision:  0.8309183856404684
# recall:  0.8322999999999999
# f1-score:  0.8303008872754171
# KNeighborsClassifier(n_neighbors=35)

# SVM
# Narrow down potential C values
param_grid = {'C': list(range(1,300,25))}
mySVM = support_vector_machine(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(mySVM.best_estimator_)

param_grid = {'C': list(range(55,65,1))}
mySVM = support_vector_machine(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(mySVM.best_estimator_)

param_grid = {'C': 61}
mySVM = support_vector_machine(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(mySVM.best_estimator_)
# c = 61
# MODEL PERFORMANCE:
# -------------
# accuracy:  0.8364
# precision:  0.8349389582000779
# recall:  0.8363999999999999
# f1-score:  0.8343713261890431
# SVC(C=61)



# RF : leaving all parameters other than n_estimators to default,
# as there should be enough samples to avoid overfitting
param_grid = {'n_estimators': list(range(1,300, 10))}
myRF = random_forest(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(myRF.best_estimator_)

param_grid = {'n_estimators': list(range(290,320, 5))}
myRF = random_forest(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(myRF.best_estimator_)

param_grid = {'n_estimators': 295}
myRF = random_forest(LDA_X, LDA_X_test, y_train, y_test, param_grid)
print(myRF.best_estimator_)

# MODEL PERFORMANCE:
# -------------
# accuracy:  0.8338
# precision:  0.8328237567998714
# recall:  0.8338
# f1-score:  0.8326965670096965
# RandomForestClassifier(n_estimators=295)



