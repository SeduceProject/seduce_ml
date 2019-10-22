from lib.learning.oracle import build_oracle, train_oracle
import numpy as np
from numpy.linalg import norm
import math
from lib.learning.knearest import KNearestOracle
from lib.learning.gaussian import GaussianProcessOracle

# DEFAULT_BATCH_SIZE = 1000
# DEFAULT_EPOCH = 300


def create_and_train_oracle(
        learning_method,
        x_train,
        y_train,
        tss_train,
        x_test,
        y_test,
        tss_test,
        epoch,
        batch_size,
        nb_inputs,
        nb_outputs,
        nb_layers,
        neurons_per_layers,
        activation_function,
        scaler=None,
        params=None,
        percentile=90):

    if params is None:
        params = {}

    plot_data = []
    score = -1

    # Build the oracle
    if learning_method == "neural":
        oracle = build_oracle(nb_inputs=nb_inputs,
                              nb_outputs=nb_outputs,
                              hidden_layers_count=nb_layers,
                              neurons_per_hidden_layer=neurons_per_layers,
                              activation_function=activation_function)

        train_oracle(oracle,
                     {
                         "x": x_train,
                         "y": y_train
                     },
                     epoch,
                     len(y_train))

        # Evaluate the neural network
        score = oracle.evaluate(x_test, y_test, batch_size=batch_size)
        print(f"score: {score}")
    elif learning_method == "knearest":
        # Build oracle using the "k-nearest neighbours" technique
        oracle = KNearestOracle(x_train, y_train, tss_train)
    elif learning_method == "gaussian":
        # Build oracle using the "gaussian process" technique
        oracle = GaussianProcessOracle(x_train, y_train, tss_train)
    else:
        raise Exception("Could not find what learning technique should be used :(")

    considered_x_set = x_test
    considered_y_set = y_test

    sum_squared_difference = 0
    for idx, e in enumerate(considered_y_set):
        test_input = np.array([considered_x_set[idx]])
        expected_value = considered_y_set[idx]

        if learning_method == "neural":
            result = oracle.predict(test_input)[0][0]
        elif learning_method == "gaussian":
            result, sigma = oracle.predict(test_input)
        else:
            result = oracle.predict(test_input)

        difference = norm(expected_value - result)

        sum_squared_difference += difference
    std = math.sqrt(sum_squared_difference / len(considered_y_set))
    print("standard deviation: %s" % std)

    differences = []

    for idx in range(0, len(considered_y_set)):
        test_input = np.array([considered_x_set[idx]])
        expected_value = considered_y_set[idx]

        if learning_method == "neural":
            result = oracle.predict(test_input)[0]
        elif learning_method == "gaussian":
            result, sigma = oracle.predict(test_input)
        else:
            result = oracle.predict(test_input)[0]

        difference = norm(expected_value - result)
        differences += [difference]

        # Scale data
        concatenated_input_and_outputs = np.array([np.append(np.copy(test_input), expected_value)])

        unscaled_expected_values = scaler.inverse_transform(concatenated_input_and_outputs)
        unscaled_predicted_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), result)]))

        expected_temp = unscaled_expected_values[:, -len(expected_value):][0]
        predicted_temp = unscaled_predicted_values[:, -len(expected_value):][0]

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

    if learning_method == "neural":
        oracle.compile(optimizer='rmsprop',
                       loss='mse')

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, percentile)].mean()

    return oracle, scaler, plot_data, rmse_perc, rmse, score


class Oracle(object):

    def __init__(self, oracle, scaler, metadata):
        self.oracle = oracle
        self.scaler = scaler
        self.metadata = metadata

    def describe_params(self):
        return self.metadata

    def predict(self, unscaled_input_values):
        # Transform 'unscaled_input_values' -> 'scaled_input_values'
        # Do a prediction from 'scaled_input_values' -> 'scaled_output_values'
        # Transform 'scaled_output_values' -> 'unscaled_output_values'
        # Return 'unscaled_ouput_values'

        return None
