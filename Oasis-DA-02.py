import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\ifood_df.csv"  # Update with the correct file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Data Preview:")
print(data.head())

# Data Exploration and Cleaning
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop missing values or fill them as needed (example: dropping rows with missing values)
data = data.dropna()

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Feature Selection for Segmentation
# Selecting relevant features for clustering
features = ['MntTotal', 'MntRegularProds', 'AcceptedCmpOverall']  # Replace with relevant columns
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Customer Segmentation using K-means
# Determine the optimal number of clusters using the elbow method
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Choosing the optimal number of clusters (for example, 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Segment'] = kmeans.fit_predict(X_scaled)

# Visualization of Customer Segments
# Perform PCA for 2D visualization if more than 2 features are used
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
data['PCA1'] = pca_components[:, 0]
data['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(14, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Segment', data=data, palette='Set1')
plt.title('Customer Segmentation Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Segment')
plt.show()

# Insights and Recommendations
segment_summary = data.groupby('Segment')[features].mean()
print("\nSegment Summary:")
print(segment_summary)

print("\nInsights and Recommendations:")
print("1. Segment customers based on average purchase value and frequency to target high-value segments.")
print("2. Personalize marketing campaigns based on customer segments to increase engagement and retention.")
print("3. Analyze underperforming segments to identify areas for improvement in customer satisfaction.")
