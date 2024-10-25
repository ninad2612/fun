# Importing necessary libraries
import pandas as pd

# Loading the Adult Census Income dataset into a DataFrame
data = pd.read_csv('adultcensusincome.csv')

# Displaying summary information about the dataset (columns, data types, non-null values)
data.info()

# Checking for missing values in each column and printing the sum per column
data.isnull().sum()

# Displaying the first 5 rows of the dataset
data.head()

# Listing all column names
data.columns

# Defining categorical columns to encode with LabelEncoder
col = ['workclass', 'education', 'marital.status', 'occupation', 
       'relationship', 'race', 'sex', 'native.country', 'income']

# Importing LabelEncoder for encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encoding each categorical column into numerical values
for c in col:
    data[c] = le.fit_transform(data[c])

# Defining the features (independent variables) and target variable
x = data[['age', 'workclass', 'fnlwgt', 'education', 'education.num', 
          'occupation', 'capital.gain', 'capital.loss', 'hours.per.week', 
          'native.country']]
y = data['income']

# Importing the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Importing train_test_split for data splitting
from sklearn.model_selection import train_test_split

# Splitting the dataset into training (80%) and testing (20%) sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Training the Random Forest model on the training data
rf.fit(xtrain, ytrain)

# Predicting the target variable for the test set
pred = rf.predict(xtest)

# Importing metrics for model evaluation
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

# Calculating and printing the accuracy of the model
acc = accuracy_score(ytest, pred)
print(acc)

# Generating and displaying the confusion matrix
cm = confusion_matrix(ytest, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Visualizing a single decision tree from the Random Forest ensemble
from sklearn import tree
import matplotlib.pyplot as plt

# Plotting the first decision tree from the Random Forest model
plt.figure(figsize=(12, 10))
tree.plot_tree(rf.estimators_[0], max_depth=5, filled=True, fontsize=10)
plt.show()
