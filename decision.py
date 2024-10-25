# Importing necessary libraries
import pandas as pd

# Loading the Adult Census Income dataset into a DataFrame
data = pd.read_csv('adultcensusincome.csv')

# Displaying summary information about the dataset (columns, data types, non-null values)
data.info()

# Checking for missing values and printing the sum per column
data.isnull().sum()

# Displaying the first 5 rows of the dataset
data.head()

# Listing all column names in the dataset
data.columns

# Defining categorical columns to be encoded using LabelEncoder
col = ['workclass', 'education', 'marital.status', 'occupation', 
       'relationship', 'race', 'sex', 'native.country', 'income']

# Importing LabelEncoder for encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encoding all specified categorical columns into numerical values
for c in col:
    data[c] = le.fit_transform(data[c])

# Defining the features (independent variables) and target variable
x = data[['age', 'workclass', 'fnlwgt', 'education', 'education.num', 
          'occupation', 'capital.gain', 'capital.loss', 'hours.per.week', 
          'native.country']]  # Features
y = data['income']  # Target variable

# Importing the DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Importing train_test_split to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Splitting the data into training (80%) and testing (20%) sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Training the Decision Tree model on the training data
dt.fit(xtrain, ytrain)

# Predicting the target variable for the test set
pred = dt.predict(xtest)

# Importing metrics and visualization tools to evaluate the model
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

# Calculating the accuracy of the model
acc = accuracy_score(ytest, pred)
print(acc)  # Printing the accuracy score

# Generating the confusion matrix for the predictions
cm = confusion_matrix(ytest, pred)

# Displaying the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Importing additional tools for visualizing the decision tree
from sklearn import tree
import matplotlib.pyplot as plt

# Plotting the decision tree (up to a maximum depth of 5) for better visualization
plt.figure(figsize=(12, 10))
tree.plot_tree(dt, max_depth=5, filled=True, fontsize=10)
plt.show()
