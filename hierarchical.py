# Importing necessary libraries
import pandas as pd

# Loading the Wholesale Customers Data into a DataFrame
data = pd.read_csv('Wholesale customers data.csv')

# Importing normalize function to scale data
from sklearn.preprocessing import normalize

# Normalizing the dataset to bring all features to a similar scale
norm = normalize(data)
norm = pd.DataFrame(norm, columns=data.columns)

# Displaying the first 5 rows of the normalized data
norm.head()

# Importing clustering and visualization tools
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Plotting the dendrogram using Ward's linkage method
plt.figure(figsize=(12, 10))
plt.title('Dendrogram')
Z = linkage(norm, method='ward')
dendrograms = dendrogram(Z)

# Adding a horizontal line to indicate potential clusters
plt.figure(figsize=(12, 10))
plt.title('Dendrogram')
plt.axhline(y=6)  # Adjust threshold for number of clusters
Z = linkage(norm, method='ward')
dendrograms = dendrogram(Z)

# Creating clusters using Agglomerative Clustering with Ward's linkage
cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
cluster.fit_predict(norm)

# Trying with complete linkage to compare results
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete')
cluster.fit_predict(norm)

# Visualizing the clusters based on 'Fresh' and 'Grocery' features
plt.figure(figsize=(10, 7))
plt.scatter(norm['Fresh'], norm['Grocery'], c=cluster.labels_)
