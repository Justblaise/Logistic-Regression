# Logistic Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

style.use('ggplot')

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#print (dataset.head())

# X is assigned the age and salary dataset
X = dataset.iloc[:, 2:4].values

# y is assigned the buy decision 
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# Also, 75% of the data will be used for training, while the remaining 25% will be for testing
# The random state is set so that everytime we run the code, the result is the same
# Random state is set 2, which is seed to the random number generator. This can any number. I have used 0 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# Use StandardScaler() to standardise the features and scaling to unit variance
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting  Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression # import the  Logistic Regression class from its lib
## Logistic Regression is used beacuse the output is binary i.e purchased or not
classifier = LogisticRegression (random_state=0)

# Fit the model according to the training data X_train,y_train.
classifier.fit(X_train,y_train)

# Predicting the Test set Result 
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix 

# this function compares the predicted results and the actual result, 
from sklearn.metrics import confusion_matrix

# typing cm + enter  in the console will show us the number of correct predictions sum them diagonally and then the incorrect predictions 
cm = confusion_matrix (y_test, y_pred) 

print (cm)


# Visualizing the training set result
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()