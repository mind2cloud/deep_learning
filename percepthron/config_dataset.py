import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# centers - количество классов
# n_features - количество признаков (столбцов)
# n_samples = количество строк в X
number_of_classes = 4

X, Y = datasets.make_blobs(n_samples=number_of_classes * 500, centers=number_of_classes, n_features=2, shuffle=True,
                           cluster_std=1, center_box=(0, 20))
# plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'g*')
# plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bo')
# plt.plot(X[:, 0][Y == 2], X[:, 1][Y == 2], 'r*')
# plt.plot(X[:, 0][Y == 3], X[:, 1][Y == 3], 'yv')
#
# plt.savefig("datasets/dataset_4.png")
# plt.close("all")

np.frombuffer(X)
np.frombuffer(Y)
dataset = open("dataset.txt", 'a')
for (a, b), e in zip(X, Y):
    dataset.write(str(a) + "," + str(b) + "," + str(e) + '\n')
