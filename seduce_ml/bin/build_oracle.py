import sys
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import dill
from sklearn.model_selection import train_test_split
from seduce_ml.data.seduce_data_loader import generate_real_consumption_data
from seduce_ml.validation.validation import validate_seduce_ml, evaluate_prediction_power
from seduce_ml.oracle.oracle import create_and_train_oracle
from seduce_ml.data.scaling import *


def train(group_by,
          percentile,
          start_date,
          end_date,
          validation_start_date,
          validation_end_date,
          tmp_figures_folder,
          shuffle,
          server_id,
          use_scaler,
          params):
    comparison_plot_data = []

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

    # for PARAMS in TEST_PARAMS:
    averages_differences = []

    best_difference = None
    best_score = None
    mse_perc = None

    i = 0

    oracles = []

    while i < MAX_ITERATION_COUNT:
        i += 1

        # Train the neural network
        consumption_data = \
            generate_real_consumption_data(start_date,
                                           end_date,
                                           data_file_path=f"data/data_{group_by}m.json",
                                           group_by=group_by,
                                           use_scaler=use_scaler,
                                           server_id=server_id,
                                           learning_method=learning_method)

        nb_inputs, nb_outputs = consumption_data.get("shape")

        train_proportion = 0.80

        x_train, x_test, y_train, y_test, tss_train, tss_test = train_test_split(consumption_data.get("x"),
                                                                                 consumption_data.get("y"),
                                                                                 consumption_data.get("tss"),
                                                                                 test_size=1 - train_proportion,
                                                                                 shuffle=shuffle)

        unscaled_x_train = unscale_input(x_train, consumption_data.get("scaler"), consumption_data.get("metadata").get("variables"))
        unscaled_x_test = unscale_input(x_test, consumption_data.get("scaler"), consumption_data.get("metadata").get("variables"))
        unscaled_y_train = unscale_output(y_train, consumption_data.get("scaler"), consumption_data.get("metadata").get("variables"))
        unscaled_y_test = unscale_output(y_test, consumption_data.get("scaler"), consumption_data.get("metadata").get("variables"))

        train_test_data = {
            "x_train": x_train,
            "unscaled_x_train": unscaled_x_train,
            "x_test": x_test,
            "unscaled_x_test": unscaled_x_test,
            "y_train": y_train,
            "unscaled_y_train": unscaled_y_train,
            "y_test": y_test,
            "unscaled_y_test": unscaled_y_test,
            "tss_train": tss_train,
            "tss_test": tss_test
        }

        consumption_data_with_train_test_data = {**consumption_data, **train_test_data}

        (oracle_object, plot_data, rmse_perc, rmse, score) =\
            create_and_train_oracle(consumption_data_with_train_test_data,
                                    learning_method=learning_method,
                                    percentile=percentile,
                                    metadata=consumption_data_with_train_test_data.get("metadata"),
                                    params=params)

        consumption_data_test = {**consumption_data}
        consumption_data_test["x"] = consumption_data_with_train_test_data.get("x_test")
        consumption_data_test["y"] = consumption_data_with_train_test_data.get("y_test")
        consumption_data_test["tss"] = consumption_data_with_train_test_data.get("tss_test")

        oracle_train_rmse, oracle_train_perc = evaluate_prediction_power(
            consumption_data_with_train_test_data,
            server_id=server_id,
            tmp_figures_folder=tmp_figures_folder,
            figure_label=f"train_predict_power_idx_{i}",
            produce_figure=False,
            oracle_object=oracle_object
        )

        oracles += [{
            "plot_data": plot_data,
            "data": consumption_data.get("data"),
            "rmse_perc": rmse_perc,
            "rmse": rmse,
            "score": score,
            "idx": i,
            "oracle_train_rmse": oracle_train_rmse,
            "oracle_train_perc": oracle_train_perc,
            "oracle_object": oracle_object
        }]

    sorted_oracles = sorted(oracles, key=lambda oracle: oracle.get("oracle_train_rmse"))
    selected_oracle = sorted_oracles[0]

    best_oracle_object = selected_oracle.get("oracle_object")
    best_plot_data = selected_oracle.get("plot_data")

    param_label = f"{{'epoch': {epoch_count}, 'nb_layers': {nb_layers}, 'neurons_per_layers': {neurons_per_layers}, 'activation_function': '{activation_function}', 'nb_inputs': {nb_inputs}, 'nb_outputs': {nb_outputs}}}"

    print("##############################################")
    print(param_label)
    print("##############################################")

    mean_difference = float(np.mean(averages_differences))

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in best_plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 80)].mean()

    print("mean_difference: %s" % (mean_difference))
    print("best_difference: %s" % (best_difference))
    print("best_score: %s" % (best_score))
    print("best_score: %s" % (best_score))
    print("best_mse_perc: %s" % (mse_perc))
    print("best_rmse: %s" % (rmse))
    print("best_rmse_perc: %s" % (rmse_perc))

    epoch = time.time()

    comparison_plot_data += [{
        "epoch": epoch,
        "nb_layers": nb_layers,
        "neurons_per_layers": neurons_per_layers,
        "activation_function": activation_function,
        "mse_perc": mse_perc,
        "rmse": rmse,
        "rmse_perc": rmse_perc,
        "tmp_figures_folder": tmp_figures_folder,
        "method": learning_method,
        "score": score
    }]

    # Draw the comparison between actual data and prediction:
    server_idx = consumption_data.get("servers_names_raw").index(server_id)

    start_step = int(0 * len(best_plot_data))
    end_step = int(1.0 * len(best_plot_data))

    sorted_plot_data = sorted(best_plot_data, key=lambda d: d["x"])[start_step:end_step]

    fig = plt.figure()
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

    plt.savefig(("%s/training_%s.pdf" % (tmp_figures_folder, param_label))
                .replace(":", " ")
                .replace("'", "")
                .replace("{", "")
                .replace("}", "")
                .replace(" ", "_")
                )

    # x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
    consumption_data_validation =\
        generate_real_consumption_data(validation_start_date,
                                       validation_end_date,
                                       data_file_path=f"data/data_validation_{group_by}m.json",
                                       group_by=group_by,
                                       scaler=consumption_data.get("scaler"),
                                       use_scaler=use_scaler,
                                       server_id=server_id,
                                       learning_method=learning_method)

    if learning_method == "neural":
        figure_label = f"validation_{learning_method}__" \
                       f"epoch_{epoch_count}__" \
                       f"layers_{nb_layers}__" \
                       f"neurons_per_layer_{neurons_per_layers}__" \
                       f"activ_{activation_function}"
    elif learning_method == "knearest":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "gaussian":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "ltsm":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "proba":
        figure_label = f"validation_{learning_method}"
    else:
        raise Exception("Could not understand which learning method is used")

    # Validate data
    validate_seduce_ml(
        consumption_data_validation,
        oracle_object=oracle_object,
        server_id=server_id,
        use_scaler=use_scaler,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=figure_label
    )

    # Evaluate prediction power
    evaluate_prediction_power(
        consumption_data_validation,
        oracle_object=oracle_object,
        server_id=server_id,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=figure_label
    )

    for oracle_data in sorted_oracles:
        this_score = oracle_data.get("score")
        oracle_idx = oracle_data.get("idx")
        this_oracle_train_rmse = oracle_data.get("oracle_train_rmse")
        this_oracle_train_perc = oracle_data.get("oracle_train_perc")

        # Evaluate prediction power
        evaluate_prediction_power(
            consumption_data_validation,
            oracle_object=oracle_object,
            server_id=server_id,
            tmp_figures_folder=tmp_figures_folder,
            figure_label=f"evaluate_rmse_{this_oracle_train_rmse:.2f}_rmse_perc{this_oracle_train_perc:.2f}_score_{this_score:.5f}_idx_{oracle_idx}",
            produce_figure=False
        )

    return best_oracle_object


if __name__ == "__main__":

    print("Hello from seduce ML")

    # epoch_count = 3000
    epoch_count = 1000
    batch_size = 1000
    group_by = 20
    percentile = 80

    network_path = "last"

    start_date = "2019-12-20T00:00:00.000Z"
    end_date = "2020-01-10T00:00:00.000Z"

    # start_date = "2019-10-23T00:00:00.000Z"
    # end_date = "2019-10-25T00:00:00.000Z"
    # end_date = "2019-09-11T00:00:00.000Z"

    validation_start_date = "2020-01-10T00:00:00.000Z"
    validation_end_date = "2020-01-16T00:00:00.000Z"

    # validation_start_date = "2019-10-21T07:00:00.000Z"
    # validation_end_date = "2019-10-23T18:00:00.000Z"

    # validation_start_date = "2019-10-20T05:00:00.000Z"
    # validation_end_date = "2019-10-21T05:00:00.000Z"

    # validation_start_date = "2019-10-23T07:00:00.000Z"
    # validation_end_date = "2019-10-20T00:00:00.000Z"
    # validation_end_date = "2019-10-27T00:00:00.000Z"
    # validation_end_date = "2019-10-22T00:00:00.000Z"
    # validation_end_date = "2019-10-27T00:00:00.000Z"

    tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))

    shuffle = True
    # shuffle = False

    server_id = "ecotype-43"

    use_scaler = True
    # use_scaler = False

    learning_method = "neural"
    # learning_method = "knearest"
    # learning_method = "gaussian"
    # learning_method = "ltsm"
    # learning_method = "proba"

    EPOCHS = [
        # 500,
        1000,
        # 2000,
        # 5000,
    ]
    NB_LAYERS = [
        3,
        # 2,
        # 4,
    ]
    NEURONS_PER_LAYER = [
        8,
        # 32,
        # 64
        # 128,
        # 256,
    ]

    ACTIVATION_FUNCTIONS = [
        # "tanh", # Should try this later
        # "relu",
        # "sigmoid",
        "linear", # Should try this later
        # "softmax",
        # "exponential"
    ]

    if learning_method in ["neural", "ltsm"]:
        MAX_ITERATION_COUNT = 1
        # MAX_ITERATION_COUNT = 10
        # MAX_ITERATION_COUNT = 15
    else:
        MAX_ITERATION_COUNT = 1

    TEST_PARAMS = [
        {
            "epoch": epoch,
            "nb_layers": nb_layers,
            "neurons_per_layers": neurons_per_layers,
            "activation_function": activation_function
        }
        for epoch in EPOCHS
        for nb_layers in NB_LAYERS
        for neurons_per_layers in NEURONS_PER_LAYER
        for activation_function in ACTIVATION_FUNCTIONS
    ]

    def compute_average_consumption(x):
        return np.mean(x)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

    # investigate_correlations(
    #     start_date,
    #     end_date,
    #     True,
    #     f"data/data_{group_by}m.json",
    #     group_by,
    #     None,
    #     True,
    #     [f"ecotype-{i}" for i in range(37, 49)],
    #     tmp_figures_folder
    # )

    for PARAMS in TEST_PARAMS:

        nb_layers = PARAMS.get("nb_layers")
        neurons_per_layers = PARAMS.get("neurons_per_layers")
        activation_function = PARAMS.get("activation_function")

        if learning_method in ["neural", "ltsm"]:
            params = {
                "epoch_count": epoch_count,
                "batch_size": batch_size,
                "nb_layers": nb_layers,
                "neurons_per_layers": neurons_per_layers,
                "activation_function": activation_function
            }
        else:
            params = {}

        oracle_object = train(
            group_by,
            percentile,
            start_date,
            end_date,
            validation_start_date,
            validation_end_date,
            tmp_figures_folder,
            shuffle,
            server_id,
            use_scaler,
            params
        )

        with open('oracle.pickle', 'wb') as oracle_object_file:
            dill.dump(oracle_object, oracle_object_file)

        with open('oracle.pickle', 'rb') as oracle_object_file:
            oracle_object = dill.load(oracle_object_file)
            prediction = oracle_object.predict(oracle_object.data.get("unscaled_x")[0])
            print(prediction)

    sys.exit(0)
