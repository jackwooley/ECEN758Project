import data_loader_utility as data_load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.neighbors import KNeighborsClassifier


train_load, val_load, test_load = data_load.get_dataloaders()
X_train, y_Train, X_test, y_test = data_load.get_tabular_data()
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


# Create classifier based on training data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data_pca, y_Train)

# Assign predictions to test data
scaled_test = pd.DataFrame(scalar.fit_transform(X_test_pd))
pca_test = pca.transform(scaled_test)
predict_test = knn.predict(pca_test)
predict_check = pd.DataFrame((y_test_arr, predict_test, y_test_arr == predict_test)).transpose()


# Measure model accuracy
accuracy = sum(predict_check[2]) / predict_check[2].count()
print(accuracy)



