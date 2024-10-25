# Importing necessary libraries
import pandas as pd

# Loading the dataset into a DataFrame
data = pd.read_csv('BostonHousingData.csv')

# Checking the number of rows in the dataset
len(data)

# Displaying the first 5 rows of the dataset
data.head()

# Displaying the summary information about the dataset (columns, data types, non-null values)
data.info()

# Checking for missing values in the dataset and printing the sum of missing values per column
print(data.isnull().sum()) 

# Filling missing values with the median of the respective columns for selected features
for i in ['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']:
    data[i] = data[i].fillna(data[i].median())

# Verifying if all missing values have been handled
print(data.isnull().sum()) 

# Separating the independent variables (features) and the target variable ('MEDV')
x = data.drop(columns=['MEDV'])  # Independent variables
y = data['MEDV']  # Target variable

# Importing required classes for model building and splitting the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Splitting the dataset into training (80%) and testing (20%) sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Initializing the Linear Regression model
lr = LinearRegression()

# Training the model on the training data
lr.fit(xtrain, ytrain)

# Predicting the target variable for the test set
pred = lr.predict(xtest)

# Importing metrics to evaluate the model's performance
from sklearn.metrics import mean_squared_error, r2_score

# Calculating Mean Squared Error (MSE) for the predictions
mse = mean_squared_error(ytest, pred)
print(mse)  # Printing the MSE

# Calculating Root Mean Squared Error (RMSE) using NumPy
import numpy as np
rmse = np.sqrt(mse) 

# Calculating the R-squared (R2) score to measure the model's performance
r2 = r2_score(ytest, pred)
r2  # Displaying the R2 score

# Visualizing the performance of the model using a scatter plot
import matplotlib.pyplot as plt

# Scatter plot of actual vs. predicted values
plt.scatter(ytest, pred)

# Plotting a red line representing the perfect prediction scenario (y = x line)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red')

# Displaying the plot
plt.show()
