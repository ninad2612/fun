# Importing necessary libraries
import pandas as pd

# Loading the Titanic dataset into a DataFrame
data = pd.read_csv('Titanic-Dataset.csv')

# Displaying the first 5 rows of the dataset
data.head()

# Displaying summary information about the dataset (columns, data types, non-null values)
data.info()

# Checking for missing values in each column and printing the sum per column
data.isnull().sum()

# Handling missing values for specific columns
for i in ['Age', 'Cabin', 'Embarked']:
    if i == 'Age':
        # Filling missing values in 'Age' with the mean age
        data[i] = data[i].fillna(data[i].mean())
    elif i == 'Cabin' or i == 'Embarked':
        # Filling missing values in 'Cabin' and 'Embarked' with the mode (most frequent value)
        data[i] = data[i].fillna(data[i].count().max())

# Verifying if missing values have been handled
data.isnull().sum()

# Displaying the first 5 rows of the dataset after filling missing values
data.head()

# Listing all column names in the dataset
data.columns

# Importing LabelEncoder for encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encoding the 'Sex' column (male/female) into numerical values (0 or 1)
data['Sex'] = le.fit_transform(data['Sex'])

# Displaying the first 5 rows after encoding the 'Sex' column
data.head()

# Selecting features (independent variables) and target (dependent variable)
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]  # Features
y = data['Survived']  # Target variable

# Importing necessary modules for data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Splitting the dataset into training (80%) and testing (20%) sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Initializing the Logistic Regression model
lg = LogisticRegression()

# Training the model on the training data
lg.fit(xtrain, ytrain)

# Predicting the target variable for the test set
pred = lg.predict(xtest)

# Importing metrics and libraries for evaluating the model
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Calculating the accuracy of the model
acc = accuracy_score(ytest, pred)
print(acc)  # Printing the accuracy score

# Creating a confusion matrix to evaluate the predictions
cm = confusion_matrix(ytest, pred)

# Displaying the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
