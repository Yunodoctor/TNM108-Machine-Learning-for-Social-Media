import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# X = np.array([[5,3],
#     [10,15],
#     [15,12],
#     [24,10],
#     [30,30],
#     [85,70],
#     [71,80],
#     [60,78],
#     [70,55],
#     [80,91] ])

# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X)

# plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
# plt.show() 

if __name__ == "__main__":
    
    customer_data = pd.read_csv('shopping_data.csv')
    print(customer_data.shape)
    print(customer_data.head())
    data = customer_data.iloc[:,3:5].values
    print(data[0:5, :])

    # view graph of data points
    labels = range(1, 200)
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(data[:,0],data[:,1], label='True Position')
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
        plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
    plt.show()
    
    # hierachical clustering
    linked = linkage(data, 'ward')
    labelList = range(1, 201)
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
    orientation='top',
    labels=labelList,
    distance_sort='descending',
    show_leaf_counts=True)
    plt.show()

    # plot clusters
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
    plt.show() 