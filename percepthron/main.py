import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# class Perceptron(object):
#
#     def __init__(self, no_of_inputs):
#         """
#         Initialises the weights with random values
#         Sets the learning rate
#         """
#         self.w = random_sample(no_of_inputs + 1)  # R, G, B + bias
#         self.lr = 0.001
#         self.bias = float(1)
#
#     def weight_adjustment(self, inputs, error):
#         """
#         Adjusts the weights in self.w
#         @param inputs a list of the input values used
#         @param error the difference between desired and calculated
#         """
#         for x in range(len(inputs)):
#             # Adjust the input weights
#             self.w[x] = self.w[x] + (self.lr * inputs[x] * error)
#
#         # Adjust the bias weight (the last weight in self.w)
#         self.w[-1] = self.w[-1] + (self.lr * error)
#
#     def result(self, inputs):
#         """
#         @param inputs one set of data
#         @returns the the sum of inputs multiplied by their weights
#         """
#         value = 0
#         for x in range(len(inputs)):
#             # Add the value of the inputs
#             value += inputs[x] * self.w[x]
#
#         # Add the value of bias
#         value += self.bias * self.w[-1]
#
#         # Put value into the SIGMOID equation
#         return float(1 / (1 + exp(-value)))
#
#     def recall(self, inputs):
#         res = self.result(inputs)
#         if res > 0.5:
#             return 'BLUE'
#         elif res <= 0.5:
#             return 'RED'
#         else:
#             return 'FAIL'


# def predict(x, w):
#     y_predicted = np.dot(x, w)
#
#     # print("predicted")
#     # print(y_predicted)
#     return y_predicted
#     # return 1.0 if y_predicted >= 0.5 else 0.0
#
# def sigmoid_activation(y_predicted):
#     sigmoid = 1 / (1 + np.exp(-y_predicted))
#     return sigmoid
#
# def hiperbolic_tan(y_predicted):
#     result = (np.exp(2 * y_predicted) - 1)/(np.exp(2 * y_predicted) + 1)
#     return result
#
# def train_weights(matrix, weights, nb_epoch=100, l_rate=0.01):
#     # l_rate = 0.01;
#     for epoch in range(0, nb_epoch):
#         cur_acc = accuracy(matrix, weights)
#         print("\nEpoch %d \nWeights: " % epoch, weights)
#         print("Accuracy: ", cur_acc)
#         # Цикл по строкам матрицы
#         for i in range(len(matrix)):
#             prediction = sigmoid_activation(predict(matrix[i][:-1], weights))
#             # error = matrix[i][-1] - prediction
#             for j in range(len(weights)):
#                 weights[j] = weights[j] + (l_rate * calc_error(matrix[i][-1], prediction) * matrix[i][j])
#
#         # weight += l_rate * calc_error(y, predict(x, weight)) * x;
#     # plt.plot(matrix, weights, title="Final Epoch")
#     return weights
#
# def define_class(prediction, y):
#     if Counter(prediction)[1] > 1:  # if there code looks like 1100, 1010... (more than one 1)
#         return False
#     if prediction[int(y)] > 0.5:  # if the place of 1 is right
#         return True
#     else:
#         return False
#
# def calc_error(y, y_pred):
#     error = y_pred - y;
#     return error
#
# def accuracy(matrix, weights):
#     num_correct = 0.0
#     preds = []
#     for i in range(len(matrix)):
#         # pred = predict(matrix[i][:-1], weights)  # get predicted classification
#         # preds.append(pred)
#         # if pred == matrix[i][-1]: num_correct += 1.0
#         predictions = []
#         for k in weights:
#             if predict(matrix[i][:-1], weights) > 0.5:
#                 predictions.append(1)
#             else:
#                 predictions.append(0)
#         if define_class(predictions, matrix[i][-1]):
#             num_correct += 1.0
#     print("Predictions:", preds)
#     print("num_correct:", num_correct)
#     return num_correct / float(len(matrix))
#
#
# if __name__ == '__main__':
#     nb_epoch = 100
#     l_rate = 0.01
#     number_of_neurons = 10
#
#     matrix = []
#     with open('titanic.dat') as f:
#         lines = f.readlines()
#         for line in lines:
#             new_line = '1.0,' + line  # append bias input for dataset
#             new_line = new_line.replace(',', ' ')
#             matrix.append([float(i) for i in new_line.split(' ')])
#     # 3 inputs (including single bias input), 3 weights
#     matrix = [[1.00, 0.08, 0.72, 1.0],
#               [1.00, 0.10, 1.00, 0.0],
#               [1.00, 0.26, 0.58, 1.0],
#               [1.00, 0.35, 0.95, 0.0],
#               [1.00, 0.45, 0.15, 1.0],
#               [1.00, 0.60, 0.30, 1.0],
#               [1.00, 0.70, 0.65, 0.0],
#               [1.00, 0.92, 0.45, 0.0]]
#
#     X_train, X_test, X_val = np.split(matrix, [int(.8 * len(matrix)), int(.9 * len(matrix))])
#     weights = np.random.uniform(-1, 1, size=(number_of_neurons, len(matrix[0]) - 1))
#
#     #weights = [0.20, 1.00, -1.00]  # initial weights specified in problem
#     result = train_weights(X_train, weights=weights, nb_epoch=nb_epoch, l_rate=l_rate)
#     print(result)
#     print(len(matrix))


class SLPerceptron(object):
    def __init__(self, l_rate, nb_epoch, number_of_neurons, weights, type_of_activation_function, dataset):
        self.l_rate = l_rate
        self.nb_epoch = nb_epoch
        self.number_of_neurons = number_of_neurons
        self.weights = weights
        self.type_of_activation_function = type_of_activation_function
        self.dataset = dataset

    def RBF(self, x, c, r):
        return np.exp(-((x - c) ** 2) / (r ** 2))

    def sigmoid(self, x):
        result = 1 / (1 + np.exp(-x))
        return result

    def define_class(self, prediction, y):
        if Counter(prediction)[1] > 1:  # if there code looks like 1100, 1010... (more than one 1)
            return False
        if (prediction[int(y)] > 0.5):  # if the place of 1 is right
            return True
        else:
            return False

    def predict(self, matrix, weights):
        output = sum([matrix[k] * weights[k] for k in range(len(matrix))])
        if self.type_of_activation_function == "RBF":
            c = 5
            sigm = np.random.normal(1, 1000)
            return self.RBF(output, c, sigm)
        if self.type_of_activation_function == "sigmoid":
            return self.sigmoid(output)

    def find_accuracy(self, matrix, weights):
        num_correct = 0.0
        for i in range(len(matrix)):
            predictions = []
            for k in weights:
                predictions.append(1 if self.predict(matrix[i][:-1], k) > 0.5 else 0)
            if self.define_class(predictions, matrix[i][-1]):
                num_correct += 1.0
        return num_correct / float(len(matrix))

    def find_error(self, matrix):
        error = []
        prediction = []
        mse = []
        code_class = np.zeros(self.number_of_neurons)
        for i in range(len(matrix)):
            error_all_neurons = 0
            code_class[int(matrix[i][-1])] = 1.0
            for k in self.weights:
                prediction.append(self.predict(matrix[i][:-1], k))
            for s in range(len(code_class)):
                error_all_neurons += abs(code_class[s] - prediction[s])
            prediction = []
            code_class[int(matrix[i][-1])] = 0.0
            error.append(error_all_neurons / self.number_of_neurons)
            mse.append(error_all_neurons ** 2 / self.number_of_neurons)
        return sum(mse) / len(matrix), sum(error) / len(matrix)

    def confusion_matrix(self, matrix, weights):
        x = []
        recall = []
        precision = []
        for i in range(len(matrix)):
            x.append(matrix[i][-1])
        count_classes = Counter(x)
        confusion_matrix = np.zeros(shape=(len(count_classes), len(count_classes)))
        for i in range(len(matrix)):
            predictions = []
            for k in weights:
                predictions.append(1 if self.predict(matrix[i][:-1], k) > 0.5 else 0)
            if Counter(predictions)[1] == 1:
                for y_pred in range(len(count_classes)):
                    if (predictions[y_pred] > 0.5):
                        y = int(matrix[i][-1])
                        confusion_matrix[y_pred][y] += 1.0
        sum_col = np.sum(confusion_matrix, axis=0)  # сумма по столбцам
        sum_str = np.sum(confusion_matrix, axis=1)  # сумма по строкам
        for i in range(len(confusion_matrix)):
            recall.append(confusion_matrix[i][i] / sum_col[i])
            precision.append(confusion_matrix[i][i] / sum_str[i])
        return recall, precision

#    def plot(self, x, y, xlabel, ylabel, label, title, color):
#        plt.plot(x, y, linestyle='-', linewidth=1, color=color, label=label)
#        plt.title(title)
#        plt.xlabel(xlabel)
#        plt.ylabel(ylabel)
#        plt.legend()
#
#    def plot_data_predicted(self, matrix, number_of_classes):
#        data_x1 = [matrix[i][1] for i in range(len(matrix))]
#        data_x2 = [matrix[i][2] for i in range(len(matrix))]
#        x_min = min(data_x1)
#        x_max = max(data_x1)
#
#        y_min = min(data_x2)
#        y_max = max(data_x2)
#
#        plt.plot(x_min, y_min, color="magenta", marker="v")
#        plt.plot(x_min, y_max, color="magenta", marker="v")
#        plt.plot(x_max, y_min, color="magenta", marker="v")
#        plt.plot(x_max, y_max, color="magenta", marker="v")
#        color = (0.0, 1.0, 0.5)
#        for i in range(len(matrix)):
#            for j in range(number_of_classes):
#                if (matrix[i][3] == j):
#                    color = (j / (j + 1), j / (j + 2), j / (j + 3))
#            plt.plot(matrix[i][1], matrix[i][2], marker="*", color=color)
#
#        for x2 in np.arange(y_min, y_max, (y_max - y_min) / 30):
#            for x1 in np.arange(x_min, x_max, (x_max - x_min) / 30):
#                predictions = []
#                for k in self.weights:
#                    predictions.append(1.0 if self.predict([1, x1, x2], k) > 0.5 else 0)
#                if Counter(predictions)[1] == 1:
#                    for j in range(number_of_classes):
#                        if (np.argmax(predictions) == j):
#                            color = (1 - j / (j + 4), 1 - j / (j + 5), j / (j + 6))
#                    plt.plot(x1, x2, marker="o", color=color, alpha=0.4)
#        plt.savefig("../results/class_regions.png")

    def fit(self, X_train, X_val):
        prediction = []
        error_full_train = []  # error for all the network on TRAIN data
        error_full_val = []  # error for all the network on VAL data
        error = []  # error for each neuron
        accuracy = []  # collect accuracy for all the epochs for the plot
        code_class = np.zeros(self.number_of_neurons)  # code the class (for instance 1000 if it is the 0 class)
        for epoch in range(self.nb_epoch):
            cur_acc = self.find_accuracy(X_train, self.weights)
            recall, precision = self.confusion_matrix(X_train, self.weights)

            mse_train, error_train = self.find_error(X_train)
            mse_val, error_val = self.find_error(X_val)

            print("\nEpoch: ", epoch, "| Accuracy: ", cur_acc, "| Recall: ", sum(recall) / len(recall), "| Precision: ",
                  sum(precision) / len(precision), "| Error_train: ", error_train, "| MSE_train: ", mse_train)

            error_full_train.append(error_train)
            error_full_val.append(error_val)

            accuracy.append(cur_acc)
            if cur_acc > 0.90: break

            for i in range(len(X_train)):
                code_class[int(X_train[i][-1])] = 1
                for k in self.weights:
                    prediction.append(self.predict(X_train[i][:-1], k))
                for s in range(len(code_class)):
                    error.append(code_class[s] - prediction[s])
                    for t in range(len(self.weights[0])):
                        self.weights[s][t] = self.weights[s][t] + (self.l_rate * error[s] * X_train[i][t])
                code_class[int(X_train[i][-1])] = 0
                prediction = []
                error = []

        # PLOT FOR ACCURACY
        x = np.array([i for i in range(epoch + 1)])
        self.plot(x, accuracy, "epoch", "accuracy", "train", "Accuracy", "darkmagenta")
        plt.savefig("../results/accuracy_plot_train.png")
        plt.close("all")

        # PLOT FOR ERRORS
        x = np.array([i for i in range(epoch + 1)])
        self.plot(x, error_full_train, "Epoch", "Error", "train data", "Running error", "green")
        self.plot(x, error_full_val, "Epoch", "Error", "val data", "Running error", "red")
        plt.savefig("../results/error/error_plot.png")
        plt.close("all")
        print("\nWeights: ", self.weights)


def main():
    nb_epoch = 100
    l_rate = 0.09
    number_of_neurons = 3
    dataset = "dataset_3"
    # Read the data from file
    matrix = []
    with open('titanic.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            new_line = '1.0,' + line  # append bias input for dataset
            new_line = new_line.replace(',', ' ')
            matrix.append([float(i) for i in new_line.split(' ')])

    np.random.shuffle(matrix)  # shuffle the data

    # Split matrix data for train, test, validation data in corresponding percentage
    # X_train, X_test, X_val = np.split(matrix, [int(.8 * len(matrix)), int(.9 * len(matrix))])
    # weights = np.random.uniform(-1, 1, size=(number_of_neurons, len(matrix[0]) - 1))
    #
    # # TRAIN ON TRAINING DATA with random weights
    # model = SLPerceptron(l_rate, nb_epoch, number_of_neurons, weights, "sigmoid", dataset)
    # model.fit(X_train, X_val)
    # model.plot_data_predicted(X_train, number_of_classes = 2)
    #
    # # Check the network on TEST DATA
    # print("Accuracy for test data: ", model.find_accuracy(X_test, model.weights))


if __name__ == '__main__':
    main()

    data = ["paris", "barcelona", "kolkata", "new york"]
    import random

    print([random.sample(data, 2) for _ in xrange(5)])