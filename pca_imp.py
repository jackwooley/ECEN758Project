import data_loader_utility as data_load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


train_load, val_load, test_load = data_load.get_dataloaders()
X_train, y_train, X_test, y_test = data_load.get_tabular_data()
X_train_pd = pd.DataFrame(X_train)
X_test_pd = pd.DataFrame(X_test)
y_test_arr = np.array(y_test)

#Standardize the features
#Create an object of StandardScaler which is present in sklearn.preprocessing
scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(X_train_pd)) #scaling the data
scaled_data


#Applying PCA
#Taking no. of Principal Components as 45
pca = PCA(n_components=45)
pca.fit(scaled_data)
print(sum(pca.explained_variance_ratio_))
data_pca = pca.transform(scaled_data)
data_pca = pd.DataFrame(data_pca)
data_pca.head()
scaled_test = pd.DataFrame(scalar.fit_transform(X_test_pd))
pca_test = pca.transform(scaled_test)


# Use tSNE for visualization
tsne = TSNE(n_components=3, learning_rate='auto',init='random', perplexity=3)
X_tsne = tsne.fit_transform(data_pca)
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X_tsne[:,0],X_tsne[:,1], X_tsne[:,2])
plt.show()


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

# Narrow down potential k values
param_grid = {'n_neighbors': list(range(1,243,6))}
knn_test = KNeighborsClassifier()
optimal = GridSearchCV(knn_test, param_grid, refit=True)
optimal.fit(data_pca.values, y_train)

# Find best k near previous result
param_grid = {'n_neighbors': list(range(1,15,2))}
knn_test = KNeighborsClassifier()
optimal = GridSearchCV(knn_test, param_grid, refit=True)
optimal.fit(data_pca.values, y_train)


# Test model built with best k (7)
# Assign predictions to test data
predict_test = optimal.predict(pca_test)
predict_check = pd.DataFrame((y_test_arr, predict_test, y_test_arr == predict_test)).transpose()


# Measure model accuracy
accuracy = sum(predict_check[2]) / predict_check[2].count()
print(accuracy)



