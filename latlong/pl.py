import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the CSV

data = pd.read_csv("labels_train.csv")

# Container for outliers

outliers = pd.DataFrame()

# Compute Z-scores and identify outliers per region

data['outlier'] = False
for region_id, group in data.groupby('Region_ID'):
    group['lat_z'] = zscore(group['latitude'])
    group['long_z'] = zscore(group['longitude'])
    group['outlier'] = (group['lat_z'].abs() > 2) | (group['long_z'].abs() > 2)
    outliers = pd.concat([outliers, group[group['outlier']]])
    data.loc[group.index, 'lat_z'] = group['lat_z']
    data.loc[group.index, 'long_z'] = group['long_z']
    data.loc[group.index, 'outlier'] = group['outlier']

# Print outlier filenames and coordinates

print("Outlier Filenames per Region:")
print(outliers[['filename', 'Region_ID', 'latitude', 'longitude']])

# Plotting

plt.figure(figsize=(10, 7))
unique_regions = data['Region_ID'].unique()
cmap = plt.get_cmap('tab10')  # Correct way

for i, region_id in enumerate(unique_regions):
    group = data[(data['Region_ID'] == region_id) & (~data['outlier'])]
    plt.scatter(group['latitude'], group['longitude'],
    label=f'Region {region_id}',
    color=cmap(i % 10))  # Wrap-around if >10 regions

# Overlay outliers in red

plt.scatter(outliers['latitude'], outliers['longitude'],
color='red', label='Outliers', edgecolor='black', marker='x', s=100)

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Latitude vs Longitude by Region with Outliers')
plt.legend()
plt.grid(True)
plt.show()
