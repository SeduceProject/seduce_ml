from sklearn.ensemble import GradientBoostingRegressor
from seduce_ml.oracle.oracle import Oracle
from seduce_ml.data.scaling import *


class GradientBoostRegressorOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def train(self, data, params):
        self.data = data

        self.ests = []

        x_train = data.scaled_train_df[data.metadata.get("input")].to_numpy()
        y_train = data.scaled_train_df[data.metadata.get("output")].to_numpy()

        for idx, output_variable_name in enumerate(self.metadata.get("output")):

                y_train_this_column = y_train[:, idx]

                est = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=1,
                    random_state=0,
                    loss='ls')
                est.fit(x_train, y_train_this_column)

                self.ests += [est]

        self.state = "TRAINED"

    def predict(self, x_input):
        rescaled_x_input = rescale_input(x_input.reshape(1, len(self.metadata.get("input"))), self.scaler, self.metadata.get("variables"))

        result = []

        for est in self.ests:
            y_pred = est.predict(rescaled_x_input)
            result += [y_pred]

        rescaled_result = unscale_output(np.array(result), self.scaler, self.metadata.get("variables"))
        return rescaled_result.reshape(1, len(self.metadata.get("output")))
