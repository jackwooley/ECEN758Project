import data_loader_utility as data_load
import pandas as pd

train_load, val_load, test_load = data_load.get_dataloaders()
X_train, y_Train, X_test, y_test = data_load.get_tabular_data()
X_train_pd = pd.DataFrame(X_train)