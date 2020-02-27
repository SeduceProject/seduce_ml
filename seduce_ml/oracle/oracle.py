import math
from numpy.linalg import norm
from seduce_ml.data.scaling import *


def create_and_train_oracle(
        consumption_data,
        learning_method,
        params=None,
        percentile=90,
        metadata=None,
        one_oracle_per_output=True):

    from seduce_ml.learning.neural import NeuralNetworkOracle
    from seduce_ml.learning.knearest import KNearestOracle
    from seduce_ml.learning.gaussian import GaussianProcessOracle
    from seduce_ml.learning.proba import ProbaOracle
    from seduce_ml.learning.gradient_boosting_regressor import GradientBoostRegressorOracle
    from seduce_ml.learning.ltsm import LstmOracle
    from seduce_ml.learning.one_oracle_per_output import OneOraclePerOutput

    if params is None:
        params = {}

    if metadata is None:
        metadata = {
            "input": [],
            "output": []
        }

    x_train = consumption_data.get("x_train")
    y_train = consumption_data.get("y_train")
    unscaled_x_train = consumption_data.get("unscaled_x_train")
    unscaled_y_train = consumption_data.get("unscaled_y_train")
    scaler = consumption_data.get("scaler")

    variables = metadata["input"] + metadata["output"]
    expected_output_columns_count = len(metadata["output"])

    plot_data = []
    score = -1

    # Create an oracle object
    new_oracle = None

    # Build the oracle
    if learning_method == "neural":
        oracle_class = NeuralNetworkOracle
    elif learning_method == "knearest":
        oracle_class = KNearestOracle
    elif learning_method == "gaussian":
        oracle_class = GaussianProcessOracle
    elif learning_method == "proba":
        oracle_class = ProbaOracle
    elif learning_method == "gradient_boost_regressor":
        oracle_class = GradientBoostRegressorOracle
    elif learning_method == "ltsm":
        oracle_class = LstmOracle

    if one_oracle_per_output:
        new_oracle = OneOraclePerOutput(scaler, metadata, params, oracle_class)
    else:
        new_oracle = oracle_class(scaler, metadata, params)

    if new_oracle is None:
        raise Exception(f"Could not understand which Oracle to use (learning_method='{learning_method}')")

    new_oracle.train(consumption_data)

    considered_y_set = y_train

    unscaled_considered_x_set = unscaled_x_train
    unscaled_considered_y_set = unscaled_y_train

    sum_squared_difference = 0
    for idx, e in enumerate(considered_y_set):
        test_input = np.array([unscaled_considered_x_set[idx]])
        expected_value = unscaled_considered_y_set[idx]

        result = new_oracle.predict(test_input)

        difference = norm(expected_value - result)

        sum_squared_difference += difference
    std = math.sqrt(sum_squared_difference / len(considered_y_set))
    print("standard deviation: %s" % std)

    differences = []

    for idx in range(0, len(considered_y_set)):
        test_input = np.array([unscaled_considered_x_set[idx]])
        expected_value = unscaled_considered_y_set[idx]

        result = new_oracle.predict(test_input)

        difference = norm(expected_value - result)

        differences += [difference]

        result = np.array(result).reshape([1, expected_output_columns_count])
        expected_value = np.array(expected_value).reshape([1, expected_output_columns_count])

        predicted_temp = unscale_output(result, scaler, variables)[0]
        expected_temp = unscale_output(expected_value, scaler, variables)[0]

        # Compute MSE
        mse = ((predicted_temp - expected_temp) ** 2)

        # Save some data for plotting
        plot_data += [{
            "x": idx,
            "y_actual": expected_value,
            "y_pred": result,
            "rmse_mean": np.sqrt(mse).mean(axis=0),
            "rmse_raw": np.sqrt(mse),
            "temp_actual": expected_temp,
            "temp_pred": predicted_temp,
        }]

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, percentile)].mean()

    return new_oracle, plot_data, rmse_perc, rmse, score


class Oracle(object):

    def __init__(self, scaler, metadata, params):
        self.scaler = scaler
        self.metadata = metadata
        self.params = params
        self.data = None
        self.state = "CREATED"

    def get_state(self):
        return self.state

    def train(self, data):
        raise Exception("Not yet implemented!")

    def predict(self, unscaled_input_values):
        raise Exception("Not yet implemented!")
