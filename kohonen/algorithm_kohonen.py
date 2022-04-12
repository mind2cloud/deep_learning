import numpy as np

class Kohonen:
    def __init__(self, dim, rows, cols, X, learn_rate_max=0.6, step_max=3000):
        self.dim = dim
        self.rows = rows
        self.cols = cols
        self.range_max = self.rows + self.cols
        self.learn_rate_max = 0.6
        self.step_max = 3000
        self.X = X

    def euclidian_distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def manhattan_distance(self, r1, c1, r2, c2):
        return np.abs(r1 - r2) + np.abs(c1 - c2)

    def get_closest_node(self, data, t, kohonen_map, m_rows, m_cols):
        result = (0, 0)
        small_dist = 1.0e20
        for i in range(m_rows):
            for j in range(m_cols):
                ed = self.euclidian_distance(kohonen_map[i][j], data[t])
                if ed < small_dist:
                    small_dist = ed
                    result = (i, j)
        return result

    def most_common(self, lst, n):
        if len(lst) == 0: return -1
        counts = np.zeros(shape=n, dtype=np.int)
        for i in range(len(lst)):
            counts[lst[i]] += 1
        return np.argmax(counts)

    def fit(self):
        kohonen_map = np.random.random_sample(size=(self.rows, self.cols, self.dim))
        for s in range(self.step_max):
            percentage = 1.0 - ((s * 1.0) / self.step_max)
            current_range = (int)(percentage * self.range_max)
            current_learning_rate = percentage * self.learn_rate_max

            t = np.random.randint(len(self.X))
            (bmu_row, bmu_col) = self.get_closest_node(self.X, t, kohonen_map, self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.manhattan_distance(bmu_row, bmu_col, i, j) < current_range:
                        kohonen_map[i][j] = kohonen_map[i][j] + current_learning_rate * (self.X[t] - kohonen_map[i][j])
        return kohonen_map