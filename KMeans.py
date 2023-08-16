import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.datasets import make_blobs


colors = ['green', 'orange', 'red']

k = 3 # количетсво кластеров

# генерация датасета с случайными признаками
X, y = make_blobs(n_samples=100, random_state=2) 

# визуализация готового датасета
plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1])
plt.show()


def KMeans(X, k=3, min_distance= 1e-4, num_iter=10):
    
    # задаем начельное положение центроидов, находим их случайним образом
    np.random.seed(10)
    centroid_ids = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[centroid_ids]

    for i in tqdm.tqdm(range(num_iter)):
        clusters = {i:[] for i in range(k)} # колекция будущих класетров
        
        # находим ближайшие точки к каждому центроиду
        for x in X:
            distances = np.linalg.norm(centroids - x, axis=1)
            cluster_ind = distances.argmin()
            clusters[cluster_ind].append(x)
        
        # находим новые координаты центроидов в готовых кластерах 
        new_centroids = {}
        for cluster in clusters:
            new_centroids[cluster] = np.mean(clusters[cluster], axis=0)
            
        new_centroids = dict(sorted(new_centroids.items()))
        new_centroids = np.array(list(new_centroids.values())) 
           
        for x in X:
            distances = np.linalg.norm(new_centroids - x, axis=1)
            cluster_ind = distances.argmin()
            clusters[cluster_ind].append(x)
            
        new_centroids = {}
        for cluster in clusters:
            new_centroids[cluster] = np.mean(clusters[cluster], axis=0)
            
        new_centroids = dict(sorted(new_centroids.items()))
        new_centroids = np.array(list(new_centroids.values())) 
            
        
        #centroids = new_centroids.copy()
        
        # проверка условия остановки итераций
        is_stop = True
        for clust in range(len(centroids)):
            if np.linalg.norm(centroids[clust] - new_centroids[clust]) > min_distance:
                is_stop = False
                break
        
        if is_stop:
            print(f'Количество итераций {i}')
            break
        
        centroids = new_centroids.copy()
        
    return centroids, clusters


centroids, clusters = KMeans(X)

# визуализация обученой модели
plt.figure(figsize=(7, 7))
for i in clusters: # задаем цвет каждому кластеру
    for x in clusters[i]:
        plt.scatter(x[0], x[1], color=colors[i])
        
for i, center in enumerate(centroids): # вывод центроидов на график
    plt.scatter(center[0], center[1], marker='*', s=400, c=colors[i])
plt.show()