import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from lib.deeplearning.oracle import build_oracle, train_oracle
from lib.data.seduce_data_loader import generate_real_consumption_data
import random
import os
import time
from texttable import Texttable
from sklearn.externals import joblib
from validate_seduce_ml import validate_seduce_ml
import uuid

EPOCH_COUNT = 3000
BATCH_SIZE = 1000
GROUP_BY = 60
PERCENTILE = 80

start_date = "2019-06-01T00:00:00.000Z"
end_date = "2019-07-05T00:00:00.000Z"

NETWORK_PATH = "last"

validation_start_date = "2019-07-05T00:00:00.000Z"
validation_end_date = "2019-07-12T18:00:00.000Z"

tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))
os.makedirs(tmp_figures_folder)


COMPARISON_PLOT_DATA = []

EPOCHS = [500, 1000, 2000]
NB_LAYERS = [1]
NEURONS_PER_LAYER = [64, 128]
ACTIVATION_FUNCTIONS = [
    "tanh",
    "relu",
    # "sigmoid",
    "linear",
    # "softmax",
    # "exponential"
]
NB_RUN = 5
SERVER_ID = 43

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


def sort_tasks_scheduler(loads):
    return sorted(loads)


def sort_tasks_scheduler(loads):
    return sorted(loads)


if __name__ == "__main__":
    print("Hello from seduce ML")

    for PARAMS in TEST_PARAMS:
        # nb_servers = 48
        nb_servers = 48

        best_data = None
        best_plot_data = []
        scores = []
        averages_differences = []
        averages_differences_after_correction = []

        best_difference = None
        best_difference_after_correction = None
        best_score = None
        best_oracle = None
        best_scaler = None

        best_mse = None
        mse_perc = None
        best_rmse = None
        best_rmse_perc = None

        for i in range(0, NB_RUN):
            # Display a message
            print("Run %s/%s" % (i, NB_RUN))

            plot_data = []

            # Train the neural network
            x, y, tss, data, scaler, shape =\
                generate_real_consumption_data(start_date,
                                               end_date,
                                               group_by=GROUP_BY)

            nb_inputs, nb_ouputs = shape

            # Build the neural network
            oracle = build_oracle(nb_inputs=nb_inputs,
                                  nb_outputs=nb_ouputs,
                                  hidden_layers_count=PARAMS.get("nb_layers"),
                                  neurons_per_hidden_layer=PARAMS.get("neurons_per_layers"),
                                  activation_function=PARAMS.get("activation_function"))

            zip_x_y = list(zip(x, y))[0:int(len(y) * 1.0)]
            random.shuffle(zip_x_y)

            random_x = np.array([t2[0] for t2 in zip_x_y])
            random_y = np.array([t2[1] for t2 in zip_x_y])
            train_proportion = 0.80
            train_probe_size = int(len(x) * train_proportion)
            x_train, y_train = random_x[len(x) - train_probe_size:], random_y[len(y) - train_probe_size:]
            x_test, y_test = random_x[:train_probe_size], random_y[:train_probe_size]

            train_oracle(oracle,
                         {
                             "x": x_train,
                             "y": y_train
                         },
                         PARAMS.get("epoch"),
                         len(y_train))

            # Evaluate the neural network
            score = oracle.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
            print("score: %s" % (score))

            # Use the neural network

            sum_squared_difference = 0
            for idx in range(0, len(y)):
                test_input = np.array([x[idx]])
                expected_value = y[idx]

                result = oracle.predict(test_input)[0][0]

                difference = norm(expected_value - result)

                sum_squared_difference += difference
            std = math.sqrt(sum_squared_difference / len(y))
            print("standard deviation: %s" % std)

            prediction_failed = 0
            differences = []
            signed_differences = []
            for idx in range(0, len(y)):
                test_input = np.array([x[idx]])
                expected_value = y[idx]
                result = oracle.predict(test_input)[0]

                difference = norm(expected_value - result)
                differences += [difference]

                concatenated_input_and_outputs = np.array([np.append(np.copy(test_input), expected_value)])

                unscaled_expected_values = scaler.inverse_transform(concatenated_input_and_outputs)
                unscaled_predicted_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), result)]))

                expected_temp = unscaled_expected_values[:, -len(expected_value):][0]
                predicted_temp = unscaled_predicted_values[:, -len(expected_value):][0]

                mse = ((predicted_temp - expected_temp)**2)

                plot_data += [{
                    "x": idx,
                    "y_actual": expected_value,
                    "y_pred": result,
                    "mse_mean": mse.mean(axis=0),
                    "mse_raw": mse,
                    "rmse_mean": np.sqrt(mse).mean(axis=0),
                    "rmse_raw": np.sqrt(mse),
                    "temp_actual": expected_temp,
                    "temp_pred": predicted_temp,

                }]

            # std = math.sqrt(sum_squared_difference / len(y))
            print("%s / %s prediction were too far from real data" % (prediction_failed, len(y)))

            average_difference = float(np.mean(differences))
            print("average difference: %s" % (average_difference))

            differences_after_correction = []

            oracle.compile(optimizer='rmsprop',
                           loss='mse')

            averages_differences += [average_difference]

            scores += [score]

            # MSE
            flatten_mse = np.array([d["mse_raw"] for d in plot_data]).flatten()
            mse = flatten_mse.mean()
            mse_perc = flatten_mse[flatten_mse > np.percentile(flatten_mse, PERCENTILE)].mean()

            # RMSE
            flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
            rmse = flatten_rmse.mean()
            rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, PERCENTILE)].mean()

            # if best_difference is None or average_difference < best_difference:
            # if best_rmse is None or rmse < best_rmse:
            if best_rmse_perc is None or rmse_perc < best_rmse_perc:
                best_difference = average_difference
                best_oracle = oracle
                best_scaler = scaler
                best_plot_data = plot_data
                best_data = data
                best_rmse_perc = rmse_perc

        print("##############################################")
        print(PARAMS)
        print("##############################################")

        mean_difference = float(np.mean(averages_differences))
        mean_score = float(np.mean(score))

        # MSE
        flatten_mse = np.array([d["mse_raw"] for d in best_plot_data]).flatten()
        mse = flatten_mse.mean()
        mse_perc = flatten_mse[flatten_mse > np.percentile(flatten_mse, 80)].mean()

        # RMSE
        flatten_rmse = np.array([d["rmse_raw"] for d in best_plot_data]).flatten()
        rmse = flatten_rmse.mean()
        rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 80)].mean()

        print("mean_difference: %s" % (mean_difference))
        print("best_difference: %s" % (best_difference))
        print("mean_score: %s" % (mean_score))
        print("best_score: %s" % (best_score))
        print("best_score: %s" % (best_score))
        print("best_mse: %s" % (mse))
        print("best_mse_perc: %s" % (mse_perc))
        print("best_rmse: %s" % (rmse))
        print("best_rmse_perc: %s" % (rmse_perc))

        epoch = time.time()
        date_str = time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime(epoch))
        neural_net_dump_path ="data/seduceml_%s.h5" % date_str

        oracle = best_oracle

        # Dump oracle
        oracle.save(neural_net_dump_path)

        # Dump scaler
        random_scaler_filename = f"scaler_{uuid.uuid4()}"
        random_scaler_path = f"data/{random_scaler_filename}"
        joblib.dump(scaler, random_scaler_path)

        COMPARISON_PLOT_DATA += [{
            "epoch": PARAMS.get("epoch"),
            "nb_layers": PARAMS.get("nb_layers"),
            "neurons_per_layers": PARAMS.get("neurons_per_layers"),
            "activation_function": PARAMS.get("activation_function"),
            "mse": mse,
            "mse_perc": mse_perc,
            "rmse": rmse,
            "rmse_perc": rmse_perc,
            "dump_path": neural_net_dump_path,
            "tmp_figures_folder": tmp_figures_folder,
            "scaler_path": random_scaler_path
        }]

        # Draw the comparison between actual data and prediction:
        # sorted_plot_data = sorted(best_plot_data, key=lambda d: d["y_actual"])

        start_step = int(0.80 * len(best_plot_data))
        end_step = int(1.0 * len(best_plot_data))

        sorted_plot_data = sorted(best_plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        x_data = [d["x"] for d in sorted_plot_data]
        y1_data = [d["temp_actual"][SERVER_ID] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][SERVER_ID] for d in sorted_plot_data]
        x_data = range(0, len(y1_data))

        ax.plot(x_data, y1_data, color='blue', label='actual temp.')
        ax.plot(x_data, y2_data, color='red', label='predicted temp.', alpha=0.5)

        plt.legend()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of ecotype-%s (deg. C)' % SERVER_ID)
        plt.title("%s" % PARAMS)

        plt.savefig(("%s/training_%s.pdf" % (tmp_figures_folder, PARAMS))
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

        # Validate data
        validate_seduce_ml(start_date=validation_start_date,
                           end_date=validation_end_date,
                           group_by=GROUP_BY,
                           comparison_plot_data=COMPARISON_PLOT_DATA,
                           server_id=SERVER_ID)

    key = "rmse"

    for activation_function in ACTIVATION_FUNCTIONS:
        for epoch in EPOCHS:
            print("#" * 80)
            print(f"# {key.upper()} EPOCH: {epoch} ACTIVATION_FUNCTION: {activation_function}")
            print("#" * 80)
            print("\n")

            table = Texttable()
            # table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['t'] + ['f' for i in range(0, len(NEURONS_PER_LAYER))])
            table.set_cols_align(["l"] + ['r' for i in range(0, len(NEURONS_PER_LAYER))])
            rows = [["hidden layers"] + ["J=%i" % i for i in sorted(NEURONS_PER_LAYER)]]
            for nb_layers in sorted(NB_LAYERS):
                row = ["K=%i" % nb_layers]
                for neurons_per_layers in sorted(NEURONS_PER_LAYER):
                    [d] = [x[f"{key}"]
                           for x in COMPARISON_PLOT_DATA
                           if x["nb_layers"] == nb_layers and x["neurons_per_layers"] == neurons_per_layers and x["epoch"] == epoch and x["activation_function"] == activation_function]
                    row += [d]
                rows += [row]

            table.add_rows(rows)
            print(table.draw())

    for activation_function in ACTIVATION_FUNCTIONS:
        for epoch in EPOCHS:
            print("#" * 80)
            print(f"# {key.upper()} PERCENTILE {PERCENTILE} EPOCH: {epoch} ACTIVATION_FUNCTION: {activation_function}")
            print("#" * 80)
            print("\n")

            table = Texttable()
            # table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['t'] + ['f' for i in range(0, len(NEURONS_PER_LAYER))])
            table.set_cols_align(["l"] + ['r' for i in range(0, len(NEURONS_PER_LAYER))])
            rows = [["hidden layers"] + ["J=%i" % i for i in sorted(NEURONS_PER_LAYER)]]
            for nb_layers in sorted(NB_LAYERS):
                row = ["K=%i" % nb_layers]
                for neurons_per_layers in sorted(NEURONS_PER_LAYER):
                    [d] = [x[f"{key}_perc"]
                           for x in COMPARISON_PLOT_DATA
                           if x["nb_layers"] == nb_layers and x["neurons_per_layers"] == neurons_per_layers and x["epoch"] == epoch and x["activation_function"] == activation_function]
                    row += [d]
                rows += [row]

            table.add_rows(rows)
            print(table.draw())

    print("\nCOMPARISON_PLOT_DATA = %s" % COMPARISON_PLOT_DATA)

    sys.exit(0)
