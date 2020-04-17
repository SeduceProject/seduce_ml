from seduce_ml.data.scaling import *


def create_and_train_oracle(
        consumption_data,
        learning_method,
        params=None,
        metadata=None):

    from seduce_ml.learning.neural import NeuralNetworkOracle
    from seduce_ml.learning.knearest import KNearestOracle
    from seduce_ml.learning.gaussian import GaussianProcessOracle
    from seduce_ml.learning.gradient_boosting_regressor import GradientBoostRegressorOracle
    from seduce_ml.learning.ltsm import LstmOracle
    from seduce_ml.learning.autogluon import AutoGluonProcessOracle

    if params is None:
        params = {}

    if metadata is None:
        metadata = {
            "input": [],
            "output": []
        }

    unscaled_x_train = consumption_data.unscaled_train_df[consumption_data.metadata.get("input")].to_numpy()
    unscaled_y_train = consumption_data.unscaled_train_df[consumption_data.metadata.get("output")].to_numpy()
    scaler = consumption_data.scaler

    variables = metadata["input"] + metadata["output"]

    score = -1

    # Create an oracle object

    # Build the oracle
    if learning_method == "neural":
        oracle_class = NeuralNetworkOracle
    elif learning_method == "knearest":
        oracle_class = KNearestOracle
    elif learning_method == "gaussian":
        oracle_class = GaussianProcessOracle
    elif learning_method == "gradient_boost_regressor":
        oracle_class = GradientBoostRegressorOracle
    elif learning_method == "ltsm":
        oracle_class = LstmOracle
    elif learning_method == "autogluon":
        oracle_class = AutoGluonProcessOracle

    new_oracle = oracle_class(scaler, metadata, params)

    if new_oracle is None:
        raise Exception(f"Could not understand which Oracle to use (learning_method='{learning_method}')")

    new_oracle.train(consumption_data, params)

    unscaled_considered_x_set = unscaled_x_train
    unscaled_considered_y_set = unscaled_y_train

    expected_values = unscaled_considered_y_set
    predicted_values = new_oracle.predict_all(unscaled_considered_x_set)

    expected_output_shape = (1, len(consumption_data.metadata.get("output")))

    predicted_temps = np.array([unscale_output(p.reshape(expected_output_shape), scaler, variables)[0]
                       for p in predicted_values])
    expected_temps = np.array([unscale_output(e.reshape(expected_output_shape), scaler, variables)[0]
                      for e in expected_values])

    plot_data = [
        {
            "x": idx,
            "y_actual": ev,
            "y_pred": pv,
            "temp_actual": et,
            "temp_pred": pt,
        }
        for idx, (ev, pv, et, pt) in enumerate(zip(expected_values, predicted_values, expected_temps, predicted_temps))
    ]

    return new_oracle, plot_data, score


class Oracle(object):

    def __init__(self, scaler, metadata, params):
        self.scaler = scaler
        self.metadata = metadata
        self.params = params
        self.data = None
        self.state = "CREATED"

    def get_state(self):
        return self.state

    def train(self, data, params):
        raise Exception("Not yet implemented!")

    def predict(self, unscaled_input_values):
        raise Exception("Not yet implemented!")

    def predict_nsteps_in_future(self, original_data, data, nsteps, n=0):
        variables_that_travels = [var for var in self.metadata.get("variables") if var.get("become") is not None]
        step_result = self.predict(data[n])

        if n < nsteps:
            for var in variables_that_travels:
                substituting_output_var_idx = self.metadata.get("output").index(var.get("name"))
                substituting_input_var_idx = self.metadata.get("input").index(var.get("become"))
                data[n + 1][substituting_input_var_idx] = step_result[0, substituting_output_var_idx]

        if n == nsteps:
            return step_result
        else:
            return self.predict_nsteps_in_future(original_data, data, nsteps, n + 1)

    def predict_all(self, unscaled_input_values_array):
        row_count = unscaled_input_values_array.shape[0]
        return np.array(
            [self.predict(unscaled_input_values)
             for unscaled_input_values in unscaled_input_values_array]
        ).reshape(row_count, len(self.metadata.get("output")))

    def _clean_past_output_values(self, rows):
        result = rows.copy()
        variables_that_travels = [var for var in self.metadata.get("variables") if var.get("become") is not None]
        # Clear previous values
        for var in variables_that_travels:
            substituting_input_var_idx = self.metadata.get("input").index(var.get("become"))
            rows[:, 1:, substituting_input_var_idx] = -1
        return result

    def predict_all_nsteps_in_future(self, rows, nsteps):
        row_count = rows.shape[0]
        rows = self._clean_past_output_values(rows)

        return np.array(
            [self.predict_nsteps_in_future(row, row.copy(), nsteps)
             for row in rows]
        ).reshape(row_count, len(self.metadata.get("output")))
