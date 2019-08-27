from keras.models import load_model
from lib.data.seduce_data_loader import generate_real_consumption_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import datetime as dt


def validate_seduce_ml(start_date, end_date, group_by, comparison_plot_data, server_id, learning_method, servers_names_raw, use_scaler,
                       x_train=None, y_train=None, tss_train=None):

    server_idx = servers_names_raw.index(server_id)
    neighbours = [i for i in range(0, len(servers_names_raw)) if i != server_idx]

    for PLOT_DATA in comparison_plot_data:

        if learning_method == "neural":
            oracle = load_model(PLOT_DATA["dump_path"])
        else:
            oracle = joblib.load(PLOT_DATA["dump_path"])

        scaler = joblib.load(PLOT_DATA["scaler_path"])

        x, y, tss, data, scaler, shape, servers_names_raw =\
            generate_real_consumption_data(start_date,
                                           end_date,
                                           data_file_path="data_validation.json",
                                           group_by=group_by,
                                           scaler=scaler,
                                           use_scaler=use_scaler)

        start_step = 0
        end_step = len(y)

        plot_data = []

        for idx in range(0, len(y)):
            test_input = np.array([x[idx]])
            expected_value = y[idx]

            # result = compute_knearest(x_train, y_train, tss_train, test_input, expected_value)
            result = oracle.predict(test_input)

            if use_scaler:
                unscaled_expected_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), expected_value)]))
                unscaled_predicted_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), result)]))

                expected_temp = unscaled_expected_values[:, -len(expected_value):][0]
                predicted_temp = unscaled_predicted_values[:, -len(expected_value):][0]

                server_power_consumption = unscaled_predicted_values[0][server_idx]
            else:
                expected_temp = expected_value
                predicted_temp = result

                server_power_consumption = result[server_idx]

            mse = ((predicted_temp - expected_temp) ** 2)

            temp_room = -1
            if len(data.get("room_temperature")) > idx:
                temp_room = data.get("room_temperature")[idx]

            plot_data += [{
                "x": idx,
                "y_actual": expected_value,
                "y_pred": result,
                "temp_room": temp_room,
                "mse_mean": mse.mean(axis=0),
                "mse_raw": mse,
                "rmse_mean": np.sqrt(mse).mean(axis=0),
                "rmse_raw": np.sqrt(mse),
                "temp_actual": expected_temp,
                "temp_pred": predicted_temp,
                "server_power_consumption": server_power_consumption,
                # "server_power_consumption_neighbours": server_power_consumption_neighbours
            }]

        # MSE
        flatten_mse = np.array([d["mse_raw"] for d in plot_data]).flatten()
        mse = flatten_mse.mean()
        mse_perc = flatten_mse[flatten_mse > np.percentile(flatten_mse, 95)].mean()

        # RMSE
        flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
        rmse = flatten_rmse.mean()
        rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 95)].mean()

        print("best_mse: %s" % (mse))
        print("best_mse_perc: %s" % (mse_perc))
        print("best_rmse: %s" % (rmse))
        print("best_rmse_perc: %s" % (rmse_perc))

        PLOT_DATA["mse"] = mse
        PLOT_DATA["mse_perc"] = mse_perc
        PLOT_DATA["rmse"] = rmse
        PLOT_DATA["rmse_perc"] = rmse_perc

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]

        dts = [dt.datetime(int(y[0]), int(y[1]), int(y[2]), int(y[3]))
               for y in [x.split("-")
                         for x in [ts.replace("T", "-").split(":")[0]
                                   for ts in tss]]]

        sorted_dts = sorted(dts)

        ax.plot(sorted_dts, y1_data, color='blue', label='actual temp.')
        ax.plot(sorted_dts, y2_data, color='red', label='predicted temp.', alpha=0.5)
        plt.legend()

        ax2 = ax.twinx()
        ax2.legend(loc=0)
        plt.legend()

        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of %s (deg. C)' % (server_id))

        # plt.show()
        PARAMS = {
            "epoch": PLOT_DATA["epoch"],
            "nb_layers": PLOT_DATA["nb_layers"],
            "neurons_per_layers": PLOT_DATA["neurons_per_layers"],
            "activation_function": PLOT_DATA["activation_function"]
        }
        tmp_figures_folder = PLOT_DATA["tmp_figures_folder"]
        plt.savefig(("%s/real_%s.pdf" % (tmp_figures_folder, PARAMS))
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )
