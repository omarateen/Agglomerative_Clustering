import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# Step 1: Generate synthetic dataset
X, y = make_blobs(n_samples=8, centers=3, random_state=1)

# Step 2: Apply Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# Step 3: Plot clustered points with labels
plt.figure(figsize=(6, 5))

# Plot points first
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100)

# Add labels
for i, (x, y_coord) in enumerate(X):
    plt.text(x + 0.1, y_coord + 0.1, str(i), fontsize=12)

plt.title("Agglomerative Clustering (3 Clusters) with Point Labels")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 4 (Optional): Show Dendrogram
linked = linkage(X, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()