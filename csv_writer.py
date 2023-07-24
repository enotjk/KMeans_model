import pandas as pd
import csv

from KMeans import clusters

cluster_dfs = []
for i, points in clusters.items():
    cluster_df = pd.DataFrame(points, columns=['X', 'Y'])
    cluster_df['Cluster'] = f'Cluster_{i+1}'
    cluster_dfs.append(cluster_df)
cluster_df = pd.concat(cluster_dfs, ignore_index=True)
print(cluster_df)

#cluster_df.to_csv('clusters.csv', index=False)
