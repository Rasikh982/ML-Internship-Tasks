

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

data = pd.read_csv("Mall_Customers.csv")

print("Dataset ka size:", data.shape)
print(data.head())
X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []  
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(list(k_range), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.savefig("elbow_plot.png")
print("\nElbow plot bana kar 'elbow_plot.png' main save kar diya gaya hai.")
print("Graph dekh kar wahan 'elbow' (mor) wala point best k hota hai.")

best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, data["Cluster"])
print(f"\nSilhouette Score (k={best_k}):", score)
print("Note: Score jitna 1 ke kareeb ho, clusters utne acha separate hain.")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data["Cluster"], cmap="viridis")
plt.title("Customer Clusters (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("clusters_pca.png")
print("Cluster visualization 'clusters_pca.png' main save kar diya gaya hai.")

print("\nCluster Profiles (average values):")
profile = data.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
print(profile)

data.to_csv("customers_with_clusters.csv", index=False)
print("\nFinal result 'customers_with_clusters.csv' main save kar diya gaya hai.")
