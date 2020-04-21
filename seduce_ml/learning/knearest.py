from seduce_ml.oracle.oracle import Oracle
from seduce_ml.data.scaling import *


class KNearestOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def train(self, data, params):
        self.data = data
        self.state = "TRAINED"

    def _distance(self, x_input, x_train):
        return np.sqrt(np.sum((x_train - x_input) ** 2, axis=1))

    def predict(self, unscaled_input_values):

        rescaled_x_input = rescale_input(unscaled_input_values.reshape(1, len(self.metadata.get("input"))), self.scaler, self.metadata.get("variables"))

        x_train = self.data.scaled_train_df[self.data.metadata.get("input")].to_numpy()
        y_train = self.data.scaled_train_df[self.data.metadata.get("output")].to_numpy()
        tss_train = self.data.scaled_df["timestamp"].to_numpy()

        points_with_distance = [t4 for t4 in
                                zip(self._distance(rescaled_x_input, x_train), x_train, y_train, tss_train)]
        close_points = [t4 for t4 in points_with_distance]
        sorted_points = sorted(close_points, key=lambda t4: t4[0])
        selected_points = [t4[-2] for t4 in sorted_points[0:self.params.get("configuration").get("knearest").get("neighbours_count")]]
        result = np.mean(selected_points, axis=0).reshape(1, len(self.metadata.get("output")))
        unscaled_result = unscale_output(result, self.scaler, self.metadata.get("variables"))
        return unscaled_result
