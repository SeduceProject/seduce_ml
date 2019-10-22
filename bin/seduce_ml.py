import sys
import uuid
import matplotlib.pyplot as plt
from lib.data.seduce_data_loader import generate_real_consumption_data
import os
from texttable import Texttable
from sklearn.externals import joblib
from lib.validation.validation import validate_seduce_ml, evaluate_prediction_power
from lib.oracle.oracle import create_and_train_oracle
from sklearn.model_selection import train_test_split
import time
import numpy as np
from lib.data.correlation import investigate_correlations


def train(epoch_count,
          batch_size,
          group_by,
          percentile,
          network_path,
          start_date,
          end_date,
          validation_start_date,
          validation_end_date,
          tmp_figures_folder,
          shuffle,
          server_id,
          use_scaler,
          nb_layers,
          neurons_per_layers,
          activation_function):
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
        x, y, tss, data, scaler, shape, servers_names_raw, metadata = \
            generate_real_consumption_data(start_date,
                                           end_date,
                                           data_file_path=f"data/data_{group_by}m.json",
                                           group_by=group_by,
                                           use_scaler=use_scaler,
                                           server_id=server_id)

        nb_inputs, nb_outputs = shape

        train_proportion = 0.80

        x_train, x_test, y_train, y_test, tss_train, tss_test = train_test_split(x,
                                                                                 y,
                                                                                 tss,
                                                                                 test_size=1 - train_proportion,
                                                                                 shuffle=shuffle)

        (oracle, scaler, plot_data, rmse_perc, rmse, score) = create_and_train_oracle(learning_method=learning_method,
                                                                                      x_train=x_train,
                                                                                      y_train=y_train,
                                                                                      tss_train=tss_train,
                                                                                      x_test=x_test,
                                                                                      y_test=y_test,
                                                                                      tss_test=tss_test,
                                                                                      scaler=scaler,
                                                                                      params=PARAMS,
                                                                                      epoch=epoch_count,
                                                                                      batch_size=batch_size,
                                                                                      percentile=percentile,
                                                                                      nb_inputs=nb_inputs,
                                                                                      nb_outputs=nb_outputs,
                                                                                      nb_layers=nb_layers,
                                                                                      neurons_per_layers=neurons_per_layers,
                                                                                      activation_function=activation_function)

        oracle_train_rmse, oracle_train_perc = evaluate_prediction_power(
            x=x_test,
            y=y_test,
            tss=tss_test,
            server_id=server_id,
            learning_method=learning_method,
            servers_names_raw=servers_names_raw,
            use_scaler=use_scaler,
            scaler=scaler,
            oracle=oracle,
            tmp_figures_folder=tmp_figures_folder,
            figure_label=f"train_predict_power_idx_{i}",
            produce_figure=False,
            # additional_variable=ADDITIONAL_VARIABLES
        )

        oracles += [{
            # best_difference = average_difference
            "oracle": oracle,
            "scaler": scaler,
            "plot_data": plot_data,
            "data": data,
            "rmse_perc": rmse_perc,
            "rmse": rmse,
            "score": score,
            "idx": i,
            "oracle_train_rmse": oracle_train_rmse,
            "oracle_train_perc": oracle_train_perc
        }]

    sorted_oracles = sorted(oracles, key=lambda oracle: oracle.get("oracle_train_rmse"))
    selected_oracle = sorted_oracles[0]

    best_oracle = selected_oracle.get("oracle")
    best_scaler = selected_oracle.get("scaler")
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
    date_str = time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime(epoch))
    neural_net_dump_path = f"data/seduceml_{date_str}.h5"

    oracle = best_oracle

    # Dump oracle
    if learning_method == "neural":
        oracle.save(neural_net_dump_path)
    else:
        joblib.dump(best_oracle, neural_net_dump_path)

    # Dump scaler
    random_scaler_filename = f"scaler_{uuid.uuid4()}"
    random_scaler_path = f"data/{random_scaler_filename}"
    joblib.dump(best_scaler, random_scaler_path)

    comparison_plot_data += [{
        "epoch": epoch,
        "nb_layers": nb_layers,
        "neurons_per_layers": neurons_per_layers,
        "activation_function": activation_function,
        "mse_perc": mse_perc,
        "rmse": rmse,
        "rmse_perc": rmse_perc,
        "dump_path": neural_net_dump_path,
        "tmp_figures_folder": tmp_figures_folder,
        "scaler_path": random_scaler_path,
        "method": learning_method,
        "score": score
    }]

    # Draw the comparison between actual data and prediction:
    server_idx = servers_names_raw.index(server_id)

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

    x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
        generate_real_consumption_data(validation_start_date,
                                       validation_end_date,
                                       data_file_path=f"data/data_validation_{group_by}m.json",
                                       group_by=group_by,
                                       scaler=scaler,
                                       use_scaler=use_scaler,
                                       # additional_variables=ADDITIONAL_VARIABLES,
                                       server_id=server_id)

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
    else:
        raise Exception("Could not understand which learning method is used")

    # Validate data
    validate_seduce_ml(x=x_valid,
                       y=y_valid,
                       tss=tss_valid,
                       server_id=server_id,
                       learning_method=learning_method,
                       servers_names_raw=servers_names_raw,
                       use_scaler=use_scaler,
                       scaler=scaler,
                       oracle_path=neural_net_dump_path,
                       tmp_figures_folder=tmp_figures_folder,
                       figure_label=figure_label)

    # Evaluate prediction power
    evaluate_prediction_power(
        x=x_valid,
        y=y_valid,
        tss=tss_valid,
        server_id=server_id,
        learning_method=learning_method,
        servers_names_raw=servers_names_raw,
        use_scaler=use_scaler,
        scaler=scaler,
        oracle_path=neural_net_dump_path,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=figure_label,
        # additional_variable=ADDITIONAL_VARIABLES
    )

    for oracle_data in sorted_oracles:
        this_oracle = oracle_data.get("oracle")
        this_scaler = oracle_data.get("scaler")
        this_score = oracle_data.get("score")
        rmse = oracle_data.get("rmse")
        rmse_perc = oracle_data.get("rmse_perc")
        oracle_idx = oracle_data.get("idx")
        this_oracle_train_rmse = oracle_data.get("oracle_train_rmse")
        this_oracle_train_perc = oracle_data.get("oracle_train_perc")

        # Evaluate prediction power
        evaluate_prediction_power(
            x=x_valid,
            y=y_valid,
            tss=tss_valid,
            server_id=server_id,
            learning_method=learning_method,
            servers_names_raw=servers_names_raw,
            use_scaler=use_scaler,
            scaler=this_scaler,
            oracle=this_oracle,
            tmp_figures_folder=tmp_figures_folder,
            figure_label=f"evaluate_rmse_{this_oracle_train_rmse:.2f}_rmse_perc{this_oracle_train_perc:.2f}_score_{this_score:.5f}_idx_{oracle_idx}",
            produce_figure=False,
            # additional_variable=ADDITIONAL_VARIABLES
        )

    return best_oracle, scaler


if __name__ == "__main__":

    print("Hello from seduce ML")

    epoch_count = 3000
    batch_size = 1000
    group_by = 10
    percentile = 80

    network_path = "last"

    start_date = "2019-08-01T00:00:00.000Z"
    end_date = "2019-08-27T11:00:00.000Z"

    validation_start_date = "2019-08-27T12:00:00.000Z"
    validation_end_date = "2019-08-31T12:00:00.000Z"

    tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))

    shuffle = False

    server_id = "ecotype-41"

    use_scaler = True
    # use_scaler = False

    learning_method = "neural"
    # LEARNING_METHOD = "knearest"
    # LEARNING_METHOD = "gaussian"

    EPOCHS = [
        # 500,
        1000,
        # 2000,
        # 5000,
    ]
    NB_LAYERS = [
        1,
        # 2,
        # 4,
    ]
    NEURONS_PER_LAYER = [
        4,
        # 32,
        # 64
        # 128,
        # 256,
    ]

    ACTIVATION_FUNCTIONS = [
        # "tanh", # Should try this later
        "relu",
        # "sigmoid",
        # "linear", # Should try this later
        # "softmax",
        # "exponential"
    ]

    if learning_method in ["neural", "gaussian"]:
        MAX_ITERATION_COUNT = 10
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

    investigate_correlations(
        start_date,
        end_date,
        True,
        f"data/data_{group_by}m.json",
        group_by,
        None,
        True,
        [f"ecotype-{i}" for i in range(1, 49)],
        tmp_figures_folder
    )

    for PARAMS in TEST_PARAMS:

        nb_layers = PARAMS.get("nb_layers")
        neurons_per_layers = PARAMS.get("neurons_per_layers")
        activation_function = PARAMS.get("activation_function")

        train(epoch_count,
              batch_size,
              group_by,
              percentile,
              network_path,
              start_date,
              end_date,
              validation_start_date,
              validation_end_date,
              tmp_figures_folder,
              shuffle,
              server_id,
              use_scaler,
              nb_layers,
              neurons_per_layers,
              activation_function)

    sys.exit(0)
