import numpy as np

NB_NEIGHBOURS = 5


class KNearestOracle(object):

    def __init__(self, x_train, y_train, tss_train):
        self.x_train = x_train
        self.y_train = y_train
        self.tss_train = tss_train

    def predict(self, x_input):
        points_with_distance = [t4 for t4 in
                                zip(np.sqrt(np.sum((self.x_train - x_input) ** 2, axis=1)), self.x_train, self.y_train, self.tss_train)]
        close_points = [t4 for t4 in points_with_distance if (np.sqrt((t4[1][0] - x_input[0][0]) ** 2)) < 2]
        sorted_points = sorted(close_points, key=lambda t4: t4[0])
        selected_points = [t4[-2] for t4 in sorted_points[0:NB_NEIGHBOURS]]
        result = np.mean(selected_points)
        return [result]
