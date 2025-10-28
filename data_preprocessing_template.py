# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1 ].values

imputer =  SimpleImputer(missing_values=np.nan, strategy="mean")

print(dataset)

imputer.fit(X[: , 1:3])
X[:, 1:3 ]= imputer.transform(X[:, 1:3])

print(X)
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)