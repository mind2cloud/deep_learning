import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import operator
r = lambda: np.random.randint(1, 100)

class KMeans:
    def __init__(self, n_c=7):
        self.n_c = n_c
        self.centr = []

        for _ in range(n_c):
            self.centr.append(C(np.array([r(), r()])))

        colors = cm.rainbow(np.linspace(0, 1, len(self.centr)))
        for i, c in enumerate(self.centr):
            c.color = colors[i]

    def fit(self, n):
        self.n_iters = 0
        fit = False
        while not fit and self.n_iters < n:
            for point in self.X:
                closest = self.centroid(point)
                closest.points.append(point)

            if len([c for c in self.centr if c.points == c.previous_points]) == self.n_c:
                fit = True
                self.upd_centr(reset=False)
            else:
                self.upd_centr()

            self.n_iters += 1

    def centroid(self, x):

        distances = {}
        for centroid in self.centr:
            distances[centroid] = np.linalg.norm(centroid.pos - x)
        closest = min(distances.items(), key=operator.itemgetter(1))[0]
        return closest

    def upd_centr(self, reset=True):
        for centroid in self.centr:
            centroid.previous_points = centroid.points
            x_cor = [x[0] for x in centroid.points]
            y_cor = [y[1] for y in centroid.points]
            try:
                centroid.pos[0] = sum(x_cor) / len(x_cor)
                centroid.pos[1] = sum(y_cor) / len(y_cor)
            except:
                pass

            if reset:
                centroid.points = []

    def generate_samples(self, samples=7550):
        self.X = [[r(), r()] for _ in range(samples)]

    def plot_img(self):
        for i, c in enumerate(self.centr):
            plt.scatter(c.pos[0], c.pos[1], marker='o', color=c.color, s=75)
            x_cors = [x[0] for x in c.points]
            y_cors = [y[1] for y in c.points]
            plt.scatter(x_cors, y_cors, marker='.', color=c.color)

        title = 'Roman K'
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        # Save result as .png image
        plt.savefig('{}.png'.format(title))
        plt.show()

class C:
    def __init__(self, pos):
        self.pos = pos
        self.points = []
        self.previous_points = []
        self.color = None




if __name__ == '__main__':

    model_k_m = KMeans(n_c=6)
    model_k_m.generate_samples()

    # Число итерациий
    model_k_m.fit(n=30000)
    model_k_m.plot_img()