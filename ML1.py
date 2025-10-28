"""To use PCA Algorithm for dimensionality reduction. You have a dataset that includes measurements for different variables on wine 
(alcohol, ash, magnesium, and so on). Apply PCA algorithm & transform this data so that most variations in the measurements of the variables are captured by a small 
number of principal components so that it is easier to distinguish between red and white wine by inspecting these principal components."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\HP\Documents\Wine.csv")
df.head()
df.isnull().sum()
df.info()
X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_mod = PCA(n_components=2)
X_pca = pca_mod.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Customer_Segment'] = y
plt.figure(figsize=(8,6))
for seg in pca_df['Customer_Segment'].unique():
subset = pca_df[pca_df['Customer_Segment'] == seg]
plt.scatter(subset['PC1'], subset['PC2'], label=f'Segment {seg}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Wine Dataset')
plt.legend()
plt.show()
