from src.seduce_ml.learning.oracle import build_oracle as build_oracle_neural
from src.seduce_ml.learning.oracle import train_oracle as train_oracle_neural
from src.seduce_ml.learning.ltsm import build_oracle as build_oracle_ltsm
from src.seduce_ml.learning.ltsm import train_oracle as train_oracle_ltsm
import numpy as np
from numpy.linalg import norm
import math
from src.seduce_ml.learning.knearest import KNearestOracle
from src.seduce_ml.learning.gaussian import GaussianProcessOracle
from src.seduce_ml.learning.ltsm import hstack, split_sequences
from src.seduce_ml.validation.validation import rescale_input, rescale_output, unscale_input, unscale_output


def create_and_train_oracle(
        consumption_data,
        learning_method,
        epoch,
        batch_size,
        nb_inputs,
        nb_outputs,
        nb_layers,
        neurons_per_layers,
        activation_function,
        params=None,
        percentile=90,
        metadata=None):

    if params is None:
        params = {}

    if metadata is None:
        metadata = {
            "input": [],
            "output": []
        }

    x_train = consumption_data.get("x_train")
    y_train = consumption_data.get("y_train")
    tss_train = consumption_data.get("tss_train")
    x_test = consumption_data.get("x_test")
    y_test = consumption_data.get("y_test")
    tss_test = consumption_data.get("tss_test")
    scaler = consumption_data.get("scaler")

    variables = metadata["input"] + metadata["output"]
    expected_output_columns_count = len(metadata["output"])

    plot_data = []
    score = -1

    # Build the oracle
    if learning_method == "neural":
        oracle = build_oracle_neural(nb_inputs=nb_inputs,
                                     nb_outputs=nb_outputs,
                                     hidden_layers_count=nb_layers,
                                     neurons_per_hidden_layer=neurons_per_layers,
                                     activation_function=activation_function)

        train_oracle_neural(oracle,
                            {
                                "x": x_train,
                                "y": y_train
                            },
                            epoch,
                            len(y_train))

        # Evaluate the neural network
        score = oracle.evaluate(x_test, y_test, batch_size=batch_size)
        print(f"score: {score}")
    elif learning_method == "ltsm":
        oracle = build_oracle_ltsm(nb_inputs=nb_inputs,
                                   nb_outputs=nb_outputs,
                                   hidden_layers_count=nb_layers,
                                   neurons_per_hidden_layer=neurons_per_layers,
                                   activation_function=activation_function)

        train_oracle_ltsm(oracle,
                          {
                              "x": x_train,
                              "y": y_train
                          },
                          epoch,
                          len(y_train))

        # # Evaluate the neural network
        # score = oracle.evaluate(x_test, y_test, batch_size=batch_size)
        # print(f"score: {score}")
    elif learning_method == "knearest":
        # Build oracle using the "k-nearest neighbours" technique
        oracle = KNearestOracle(x_train, y_train, tss_train)
    elif learning_method == "gaussian":
        # Build oracle using the "gaussian process" technique
        oracle = GaussianProcessOracle(x_train, y_train, tss_train, scaler, metadata)
    else:
        raise Exception("Could not find what learning technique should be used :(")

    if learning_method == "ltsm":
        _, n_steps, n_features = oracle.input_shape
        data_size, _ = x_test.shape

        x = x_test.reshape(data_size, n_features)
        y = y_test.reshape(data_size, 1)

        dataset = hstack((x, y))

        x_test, y_test = split_sequences(dataset, n_steps)

    considered_x_set = x_train
    considered_y_set = y_train

    sum_squared_difference = 0
    for idx, e in enumerate(considered_y_set):
        test_input = np.array([considered_x_set[idx]])
        expected_value = considered_y_set[idx]

        if learning_method == "neural":
            result = oracle.predict(test_input)[0][0]
        elif learning_method == "gaussian":
            result, sigma = oracle.predict(test_input, idx)
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
        elif learning_method == "ltsm":
            result = oracle.predict(test_input)[0]
        elif learning_method == "gaussian":
            result, sigma = oracle.predict(test_input)
        else:
            result = oracle.predict(test_input)[0]

        difference = norm(expected_value - result)

        pass
        differences += [difference]

        # Scale data

        # concatenated_input_and_outputs = np.array([np.append(np.copy(test_input), expected_value)])
        #
        # unscaled_expected_values = scaler.inverse_transform(concatenated_input_and_outputs)
        # unscaled_predicted_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), result)]))
        #
        # expected_temp = unscaled_expected_values[:, -len(expected_value):][0]
        # predicted_temp = unscaled_predicted_values[:, -len(expected_value):][0]

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

    if learning_method == "neural":
        oracle.compile(optimizer='rmsprop',
                       loss='mse')

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, percentile)].mean()

    # Create an oracle object
    new_oracle = Oracle(oracle, scaler, metadata, learning_method, consumption_data)

    return new_oracle, plot_data, rmse_perc, rmse, score


class Oracle(object):

    def __init__(self, oracle, scaler, metadata, learning_method, data):
        self.oracle = oracle
        self.scaler = scaler
        self.metadata = metadata
        self.learning_method = learning_method
        self.data = data

    def describe_params(self):
        return self.metadata

    def predict(self, unscaled_input_values):
        # Transform 'unscaled_input_values' -> 'scaled_input_values'
        # Do a prediction from 'scaled_input_values' -> 'scaled_output_values'
        # Transform 'scaled_output_values' -> 'unscaled_output_values'
        # Return 'unscaled_ouput_values'
        scaled_input_values = rescale_input(unscaled_input_values.reshape(1, len(self.metadata.get("input"))),
                                            self.scaler,
                                            self.metadata.get("variables"))

        raw_result = self.oracle.predict(scaled_input_values)
        rescaled_result = unscale_output(raw_result, self.scaler, self.metadata.get("variables"))
        return rescaled_result.reshape(1, len(self.metadata.get("output")))
