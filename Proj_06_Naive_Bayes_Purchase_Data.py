# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
purchaseData = pd.read_csv('Purchase_Logistic.csv')

# Selecting the independent and dependent variables
X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# Create a Naive Bayes classifier and train it
nb_classifier = GaussianNB()
nb_classifier.fit(Xtrain, Ytrain)

# Predict the values using the trained classifier
Ypred = nb_classifier.predict(Xtest)

# Generate and display the confusion matrix
cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix =\n', cmat)

# Plot the original data
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1, which='both')
plt.axis('tight')
plt.show()

# Predict the colors for the data points
col = nb_classifier.predict(X)

# Plot the Naive Bayes predicted data
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=col)
plt.suptitle('Naive Bayes Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1, which='both')
plt.axis('tight')
plt.show()