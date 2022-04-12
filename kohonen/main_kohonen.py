import numpy as np
import matplotlib.pyplot as plt
import kohonen.algorithm_kohonen
from numpy import genfromtxt
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score

def main(scenario):
    """--Сценарий запуска для dataset_1--"""
    data_x = genfromtxt('../data/dataset_1.csv', delimiter=',', dtype=float, skip_header=True, usecols=(0, 1))
    data_y = genfromtxt('../data/dataset_1.csv', delimiter=',', dtype=int, skip_header=True, usecols=(2))
    attributes = len(data_x[0])
    number_of_classes = 3

    kohonen = kohonen.algorithm_kohonen.Kohonen(dim=attributes, rows=30, cols=30, X=data_x)  # dim - по количеству аттрибутов
    map = som.construct_som_map()

    print("Associating each data label to one map node ")
    clustering = np.empty(shape=(som.rows, som.cols), dtype=object)
    for i in range(som.rows):
        for j in range(som.cols):
            clustering[i][j] = []
    y_pred = []
    for t in range(len(data_x)):
        (m_row, m_col) = som.get_closest_node(data_x, t, map, som.rows, som.cols)
        cluster = m_row * som.rows + m_col
        y_pred.append(cluster)
        clustering[m_row][m_col].append(data_y[t])

    label_map = np.zeros(shape=(som.rows, som.cols), dtype=np.int)
    for i in range(som.rows):
        for j in range(som.cols):
            label_map[i][j] = som.most_common(clustering[i][j], number_of_classes)  # 3 - количество классов

    plt.imshow(label_map, cmap=plt.cm.get_cmap('nipy_spectral', number_of_classes + 1))
    plt.colorbar()
    #plt.show()
    plt.savefig('../SOM_Kohonen/results/'+str(scenario)+'.png')

    """---Вывод метрик оценки кластеризации---"""
    print("Индекс Калинского - Харабаза:", calinski_harabaz_score(data_x, y_pred))
    print("Индекс Дэвиса - Болдина:", davies_bouldin_score(data_x, y_pred))
    print("Силуэт:", silhouette_score(data_x, y_pred))
    print("Скорректированная оценка:", adjusted_rand_score(data_y, y_pred))

if __name__ == "__main__":
    """
    1 - dataset_1
    2 - dataset_2
    3 - dataset_3
    4 - dataset_4
    """
    print("Dataset 1")
    main(1)
    print("Dataset 2")
    main(2)
    print("Dataset 3")
    main(3)
    print("Dataset 4")
    main(4)
