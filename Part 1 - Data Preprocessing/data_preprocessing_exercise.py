# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Part 1 - Data Preprocessing/Data.csv')

# Separate the independent and dependent variables
# the independet variable matrix
X = dataset.iloc[:, :-1].values
# the dependet variable vector
y = dataset.iloc[:,3].values

# We have some missing data, instead of throwing the rows out, we calculate the mean of the feature (columnn) for them
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
# we only have two columns with missing data 1,2 ( 1:3 because the upper is excluded)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# For the equations we have to encode the categorical variables to numbers - 'country', 'purchased' are categorical
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

# From here we make a matrix where each column represents only one category and 0 or 1 value
# we need this, because earlier each category had different numbers, but we don't won't to have size difference
# between the categories: germany < france has no meaning here
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# for the independent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# for the test we don't need the fit, because or mean and standard deviation is calculated on the training set
X_test = sc_X.transform(X_test)