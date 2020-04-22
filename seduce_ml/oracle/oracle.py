import matplotlib.pyplot as plt
import os
import time
from seduce_ml.data.data_from_api import generate_real_consumption_data
from seduce_ml.validation.validation import evaluate_prediction_power
from seduce_ml.data.scaling import *
from abc import abstractmethod


def build_new_oracle(params, tmp_figures_folder, server_id):
    comparison_plot_data = []

    learning_method = params.get("seduce_ml").get("learning_method")

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

    oracles = []

    # Train the neural network
    consumption_data = \
        generate_real_consumption_data(params.get("seduce_ml").get("start_date"),
                                       params.get("seduce_ml").get("end_date"),
                                       data_folder_path=f"data",
                                       group_by=params.get('seduce_ml').get('group_by'),
                                       use_scaler=params.get("seduce_ml").get("use_scaler"),
                                       server_id=server_id,
                                       learning_method=learning_method)

    consumption_data.load_data()
    consumption_data.split_train_and_test_data()

    (oracle_object, plot_data, score) =\
        create_and_train_oracle(consumption_data,
                                learning_method=learning_method,
                                metadata=consumption_data.metadata,
                                params=params)

    evaluate_prediction_power(
        consumption_data,
        oracle_object=oracle_object,
        server_id=server_id,
        params=params,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=f"train_predict_power_idx",
        produce_figure=False
    )

    oracles += [{
        "plot_data": plot_data,
        "score": score,
        # "idx": i,
        "oracle_object": oracle_object
    }]

    sorted_oracles = sorted(oracles, key=lambda oracle: oracle.get("oracle_train_rmse"))
    selected_oracle = sorted_oracles[0]

    best_oracle_object = selected_oracle.get("oracle_object")
    best_plot_data = selected_oracle.get("plot_data")

    param_label = f"{learning_method}"

    epoch = time.time()

    comparison_plot_data += [{
        "epoch": epoch,
        "nb_layers": params.get("nb_layers"),
        "neurons_per_layers": params.get("neurons_per_layers"),
        "activation_function": params.get("activation_function"),
        "tmp_figures_folder": tmp_figures_folder,
        "method": learning_method,
        "score": score
    }]

    # Draw the comparison between actual data and prediction:
    server_idx = consumption_data.servers_hostnames.index(server_id)

    start_step = int(0 * len(best_plot_data))
    end_step = int(1.0 * len(best_plot_data))

    sorted_plot_data = sorted(best_plot_data, key=lambda d: d["x"])[start_step:end_step]

    # fig = plt.figure()
    ax = plt.axes()

    x_data = [d["x"] for d in sorted_plot_data]
    y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
    y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]
    x_data = range(0, len(y1_data))

    ax.plot(x_data, y1_data, color='blue', label='actual temp.', linewidth=0.5)
    ax.plot(x_data, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)

    plt.legend()

    plt.xlabel('Time (hour)')
    plt.ylabel('Back temperature of %s (deg. C)' % server_id)
    plt.title("%s" % param_label)

    plt.savefig(f"{tmp_figures_folder}/{server_id}_training_{param_label}.pdf"
                .replace(":", " ")
                .replace("'", "")
                .replace("{", "")
                .replace("}", "")
                .replace(" ", "_")
                )

    # x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
    consumption_data_validation =\
        generate_real_consumption_data(params.get('seduce_ml').get('validation_start_date'),
                                       params.get('seduce_ml').get('validation_end_date'),
                                       data_folder_path=f"data",
                                       data_file_name="data_validation.json",
                                       group_by=params.get('seduce_ml').get('group_by'),
                                       scaler=consumption_data.scaler,
                                       use_scaler=params.get('seduce_ml').get('use_scaler'),
                                       server_id=server_id,
                                       learning_method=learning_method)

    if learning_method == "neural":
        figure_label = f"validation_{learning_method}__" \
                       f"epoch_{ params.get('configuration').get('neural').get('epoch_count') }__" \
                       f"layers_{ params.get('configuration').get('neural').get('layers_count') }__" \
                       f"neurons_per_layer_{ params.get('configuration').get('neural').get('neurons_per_layers') }__" \
                       f"activ_{ params.get('configuration').get('neural').get('activation_function') }"
    elif learning_method == "knearest":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "gaussian":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "ltsm":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "gradient_boost_regressor":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "autogluon":
        figure_label = f"validation_{learning_method}"
    else:
        raise Exception("Could not understand which learning method is used")

    consumption_data_validation.load_data()

    # Evaluate prediction power
    evaluate_prediction_power(
        consumption_data_validation,
        oracle_object=oracle_object,
        server_id=server_id,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=figure_label,
        params=params
    )

    for oracle_data in sorted_oracles:
        this_score = oracle_data.get("score")
        oracle_idx = oracle_data.get("idx")

        # Evaluate prediction power
        evaluate_prediction_power(
            consumption_data_validation,
            oracle_object=oracle_object,
            server_id=server_id,
            tmp_figures_folder=tmp_figures_folder,
            figure_label=f"evaluate_score_{this_score:.5f}_idx_{oracle_idx}",
            produce_figure=False
        )

    return best_oracle_object


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

    @abstractmethod
    def train(self, data, params):
        raise Exception("Not yet implemented!")

    @abstractmethod
    def predict(self, unscaled_input_values):
        raise Exception("Not yet implemented!")

    def predict_nsteps_in_future(self, original_data, data, nsteps, n=0):

        if n == 0:
            original_data = self._clean_past_output_values(original_data)
            data = self._clean_past_output_values(data)

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

            if len(rows.shape) == 3:
                result[:, 1:, substituting_input_var_idx] = -1
            else:
                result[1:, substituting_input_var_idx] = -1
        return result

    def predict_all_nsteps_in_future(self, rows, nsteps):
        row_count = rows.shape[0]
        rows = self._clean_past_output_values(rows)

        return np.array(
            [self.predict_nsteps_in_future(row, row.copy(), nsteps)
             for row in rows]
        ).reshape(row_count, len(self.metadata.get("output")))
