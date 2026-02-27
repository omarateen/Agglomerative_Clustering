import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Generate synthetic dataset
X, y = make_blobs(random_state=1)

# Step 2: Apply Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# Step 3: Plot clustered points with labels
plt.figure(figsize=(6, 5))

# Plot points first
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100)

plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# Step 4: Show Dendrogram
linked = linkage(X, method='ward')

plt.figure(figsize=(9, 6))
dendrogram(linked)


plt.title("Dendrogram with(Ward)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.legend()
plt.show()