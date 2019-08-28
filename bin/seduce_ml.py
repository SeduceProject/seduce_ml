import sys
import uuid
import numpy as np
import matplotlib.pyplot as plt
from lib.data.seduce_data_loader import generate_real_consumption_data
import os
from texttable import Texttable
from sklearn.externals import joblib
from lib.validation.validation import validate_seduce_ml
from lib.evaluation.evaluation import evaluate
from sklearn.model_selection import train_test_split
from bin.params import *

if __name__ == "__main__":
    print("Hello from seduce ML")

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

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

        i = 0
        while i < MAX_ITERATION_COUNT:
            i += 1

            # Train the neural network
            x, y, tss, data, scaler, shape, servers_names_raw = \
                generate_real_consumption_data(start_date,
                                               end_date,
                                               group_by=GROUP_BY,
                                               use_scaler=USE_SCALER,
                                               additional_variables=ADDITIONAL_VARIABLES)

            nb_inputs, nb_outputs = shape

            PARAMS["nb_inputs"] = nb_inputs
            PARAMS["nb_outputs"] = nb_outputs

            train_proportion = 0.80

            x_train, x_test, y_train, y_test, tss_train, tss_test = train_test_split(x,
                                                                                     y,
                                                                                     tss,
                                                                                     test_size=1 - train_proportion,
                                                                                     shuffle=False)

            (oracle, scaler, plot_data, rmse_perc, rmse) = evaluate(learning_method=LEARNING_METHOD,
                                                                    x_train=x_train,
                                                                    y_train=y_train,
                                                                    tss_train=tss_train,
                                                                    x_test=x_test,
                                                                    y_test=y_test,
                                                                    tss_test=tss_test,
                                                                    scaler=scaler,
                                                                    params=PARAMS,
                                                                    percentile=PERCENTILE)

            if best_rmse is None or rmse < best_rmse:
                # best_difference = average_difference
                best_oracle = oracle
                best_scaler = scaler
                best_plot_data = plot_data
                best_data = data
                best_rmse_perc = rmse_perc
                best_rmse = rmse

        print("##############################################")
        print(PARAMS)
        print("##############################################")

        mean_difference = float(np.mean(averages_differences))

        # RMSE
        flatten_rmse = np.array([d["rmse_raw"] for d in best_plot_data]).flatten()
        rmse = flatten_rmse.mean()
        rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 80)].mean()

        print("mean_difference: %s" % (mean_difference))
        print("best_difference: %s" % (best_difference))
        # print("mean_score: %s" % (mean_score))
        print("best_score: %s" % (best_score))
        print("best_score: %s" % (best_score))
        # print("best_mse: %s" % (mse))
        print("best_mse_perc: %s" % (mse_perc))
        print("best_rmse: %s" % (rmse))
        print("best_rmse_perc: %s" % (rmse_perc))

        epoch = time.time()
        date_str = time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime(epoch))
        neural_net_dump_path = f"data/seduceml_{date_str}.h5"

        oracle = best_oracle

        # Dump oracle
        if LEARNING_METHOD == "neural":
            oracle.save(neural_net_dump_path)
        else:
            joblib.dump(best_oracle, neural_net_dump_path)

        # Dump scaler
        random_scaler_filename = f"scaler_{uuid.uuid4()}"
        random_scaler_path = f"data/{random_scaler_filename}"
        joblib.dump(best_scaler, random_scaler_path)

        COMPARISON_PLOT_DATA += [{
            "epoch": PARAMS.get("epoch"),
            "nb_layers": PARAMS.get("nb_layers"),
            "neurons_per_layers": PARAMS.get("neurons_per_layers"),
            "activation_function": PARAMS.get("activation_function"),
            # "mse": mse,
            "mse_perc": mse_perc,
            "rmse": rmse,
            "rmse_perc": rmse_perc,
            "dump_path": neural_net_dump_path,
            "tmp_figures_folder": tmp_figures_folder,
            "scaler_path": random_scaler_path,
            "method": LEARNING_METHOD
        }]

        # Draw the comparison between actual data and prediction:
        server_idx = servers_names_raw.index(SERVER_ID)

        start_step = int(0 * len(best_plot_data))
        end_step = int(1.0 * len(best_plot_data))

        sorted_plot_data = sorted(best_plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        x_data = [d["x"] for d in sorted_plot_data]
        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]
        x_data = range(0, len(y1_data))

        ax.plot(x_data, y1_data, color='blue', label='actual temp.')
        ax.plot(x_data, y2_data, color='red', label='predicted temp.', alpha=0.5)

        plt.legend()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of %s (deg. C)' % SERVER_ID)
        plt.title("%s" % PARAMS)

        plt.savefig(("%s/training_%s.pdf" % (tmp_figures_folder, PARAMS))
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

        x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw = \
            generate_real_consumption_data(validation_start_date,
                                           validation_end_date,
                                           data_file_path="data/data_validation_60m.json",
                                           group_by=GROUP_BY,
                                           scaler=scaler,
                                           use_scaler=USE_SCALER,
                                           additional_variables=ADDITIONAL_VARIABLES)

        if LEARNING_METHOD == "neural":
            figure_label = f"validation_{LEARNING_METHOD}__" \
                           f"epoch_{PARAMS.get('epoch')}__" \
                           f"layers_{PARAMS.get('nb_layers')}__" \
                           f"neurons_per_layer_{PARAMS.get('neurons_per_layers')}__" \
                           f"activ_{PARAMS.get('activation_function')}"
        elif LEARNING_METHOD == "knearest":
            figure_label = f"validation_{LEARNING_METHOD}"
        else:
            raise Exception("Could not understand which learning method is used")

        # Validate data
        validate_seduce_ml(x=x_valid,
                           y=y_valid,
                           tss=tss_valid,
                           server_id=SERVER_ID,
                           learning_method=LEARNING_METHOD,
                           servers_names_raw=servers_names_raw,
                           use_scaler=USE_SCALER,
                           scaler=scaler,
                           oracle_path=neural_net_dump_path,
                           tmp_figures_folder=tmp_figures_folder,
                           figure_label=figure_label)

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
