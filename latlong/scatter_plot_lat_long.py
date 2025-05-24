import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the CSV
data = pd.read_csv("labels_train.csv")

# Region to plot
selected_region_id = 15 # 🔁 Change this value as needed

# Filter for selected region
region_data = data[data['Region_ID'] == selected_region_id].copy()

# Extract latitude and longitude for clustering
coords = region_data[['latitude', 'longitude']].values

# Standardize the coordinates (important for DBSCAN to handle scale differences)
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# Step 1: Estimate eps using k-nearest neighbor distances
min_samples = 5  # Keep this as is or adjust based on data density
k = min_samples
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(coords_scaled)
distances, _ = neigh.kneighbors(coords_scaled)
distances = np.sort(distances[:, k-1])  # Distance to k-th nearest neighbor

# Plot k-nearest neighbor distances to guide eps selection
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.xlabel('Point Index (sorted)')
plt.ylabel(f'Distance to {k}-th Nearest Neighbor')
plt.title('K-Nearest Neighbor Distances for eps Estimation')
plt.grid(True)
plt.show()

# Step 2: Iteratively increase eps until exactly one cluster is formed
eps = 0.3  # Starting value
step = 0.1  # Increment for eps
max_eps = 2.0  # Maximum eps to try
target_clusters = 1  # Desired number of clusters

while eps <= max_eps:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
    labels = dbscan.labels_  # Cluster labels; -1 indicates outliers
    n_clusters = len(np.unique(labels[labels != -1]))  # Number of clusters (excluding outliers)
    
    print(f"Trying eps={eps:.2f}: {n_clusters} clusters")
    
    if n_clusters <= target_clusters:
        break  # Stop if we have 1 or 0 clusters
    eps += step

# Check if we achieved the goal
if n_clusters != target_clusters:
    print(f"Could not achieve exactly {target_clusters} cluster(s). Using eps={eps:.2f} with {n_clusters} clusters.")
else:
    print(f"Found exactly {target_clusters} cluster(s) with eps={eps:.2f}.")

# Add cluster labels and outlier status to the dataframe
region_data['cluster'] = labels
region_data['outlier'] = region_data['cluster'] == -1

# Get outliers
outliers = region_data[region_data['outlier']]

# Print outlier filenames and coordinates
print(f"\nOutlier Filenames for Region {selected_region_id}:")
if not outliers.empty:
    print(outliers[['filename', 'latitude', 'longitude']])
else:
    print("No outliers detected.")

# Plotting
plt.figure(figsize=(10, 7))
# Plot non-outliers (points in clusters)
for cluster_id in np.unique(labels):
    if cluster_id != -1:  # Exclude outliers
        mask = labels == cluster_id
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Cluster {cluster_id}', alpha=0.6)

# Plot outliers
if outliers.shape[0] > 0:
    plt.scatter(outliers['latitude'], outliers['longitude'], 
                color='red', label='Outliers', edgecolor='black', marker='x', s=100)

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title(f'Latitude vs Longitude - Region {selected_region_id} with DBSCAN Outliers (eps={eps:.2f})')
plt.legend()
plt.grid(True)
plt.show()

# Summary statistics
print("\nSummary:")
print(f"Total points: {len(region_data)}")
print(f"Number of outliers: {len(outliers)}")
print(f"Number of clusters: {n_clusters}")