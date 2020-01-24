import numpy as np
from src.seduce_ml.data.seduce_data_loader import get_additional_variables

NB_NEIGHBOURS = 50


class KNearestOracle(object):

    def __init__(self, x_train, y_train, tss_train):
        self.x_train = x_train
        self.y_train = y_train
        self.tss_train = tss_train
        self.variables = get_additional_variables("server-42", learning_method="knearest")
        self.weights = [1.0 / var.get("weight", 1) for var in self.variables if not var.get("output", False)]

    def distance(self, x_input):
        return np.sqrt(np.sum((self.x_train - x_input) ** 2, axis=1))

    def custom_distance(self, x_input):
        return np.sqrt(np.sum(((self.x_train-x_input) * np.array(self.weights)) ** 2, axis=1))

    def predict(self, x_input):
        points_with_distance = [t4 for t4 in
                                zip(self.distance(x_input), self.x_train, self.y_train, self.tss_train)]
        close_points = [t4 for t4 in points_with_distance if (np.sqrt((t4[1][0] - x_input[0][0]) ** 2)) < 2]
        sorted_points = sorted(close_points, key=lambda t4: t4[0])
        selected_points = [t4[-2] for t4 in sorted_points[0:NB_NEIGHBOURS]]
        result = np.mean(selected_points, axis=0)
        return [[result]]
