#ML-4
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"] = "1"

# 1️⃣ Load the dataset
df = pd.read_csv("C:\\Users\\Public\\Iris.csv")
print("Dataset preview:\n", df.head())

# Drop non-numeric or ID columns if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Extract only the feature columns (ignore 'Species' for unsupervised learning)
X = df.select_dtypes(include=['float64', 'int64'])

# 2️⃣ Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Determine the optimal number of clusters using the Elbow Method
wcss = []  # within-cluster sum of squares
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
#ML-4
# Plot the Elbow curve
plt.figure(figsize=(8,5))
plt.plot(K, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# 4️⃣ From the Elbow graph, choose optimal k (usually 3 for Iris)
optimal_k = 3
print(f"\nOptimal number of clusters (k) = {optimal_k}")

# 5️⃣ Apply KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataset
df['Cluster'] = clusters

# 6️⃣ Evaluate clustering quality using silhouette score
sil_score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {sil_score:.3f}")

# 7️⃣ Visualize the clusters using first two features
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='Set2', s=70)
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# 8️⃣ (Optional) Compare clusters with actual species
if 'Species' in df.columns:
    comparison = pd.crosstab(df['Species'], df['Cluster'])
    print("\nCluster vs Actual Species Comparison:\n", comparison)
