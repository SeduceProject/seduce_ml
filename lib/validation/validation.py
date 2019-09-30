from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import datetime as dt
import random
from bin.params import ADDITIONAL_VARIABLES


def validate_seduce_ml(x, y, tss, server_id, learning_method, servers_names_raw, use_scaler, oracle_path=None, oracle=None, scaler=None, scaler_path=None, tmp_figures_folder=None, figure_label=""):

    server_idx = servers_names_raw.index(server_id)

    if oracle is None:
        if oracle_path is None:
            raise Exception("Please provide one of those arguments: ['oracle', 'oracle_path']")
        if learning_method == "neural":
            oracle = load_model(oracle_path)
        else:
            oracle = joblib.load(oracle_path)

    if scaler is None:
        if scaler_path is None:
            raise Exception("Please provide one of those arguments: ['scaler', 'scaler_path']")
        scaler = joblib.load(scaler_path)

    start_step = 0
    end_step = len(y)

    plot_data = []

    for idx, _ in enumerate(y):
        test_input = np.array([x[idx]])
        expected_value = y[idx]

        if learning_method == "gaussian":
            result, sigma = oracle.predict(test_input)
        else:
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

        # temp_room = -1
        # if len(data.get("room_temperature")) > idx:
        #     temp_room = data.get("room_temperature")[idx]

        plot_data += [{
            "x": idx,
            "y_actual": expected_value,
            "y_pred": result,
            # "temp_room": temp_room,
            "mse_mean": mse.mean(axis=0),
            "mse_raw": mse,
            "rmse_mean": np.sqrt(mse).mean(axis=0),
            "rmse_raw": np.sqrt(mse),
            "temp_actual": expected_temp,
            "temp_pred": predicted_temp,
            "server_power_consumption": server_power_consumption,
        }]

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 95)].mean()

    if tmp_figures_folder is not None:

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]

        # dts = [dt.datetime(int(y[0]), int(y[1]), int(y[2]), int(y[3]))
        #        for y in [x.split("-")
        #                  for x in [ts.replace("T", "-").split(":")[0]
        #                            for ts in tss]]]

        dts = [dt.datetime(*[int(x)
                             for x in [a.split("-") + b.split(":")
                                       for (a, b) in [ts.replace("Z", "").split("T")]][0]])
               for ts in tss]

        sorted_dts = sorted(dts)

        ax.plot(sorted_dts, y1_data, color='blue', label='actual temp.', linewidth=0.5)
        ax.plot(sorted_dts, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)
        plt.legend()

        ax2 = ax.twinx()
        ax2.legend(loc=0)
        plt.legend()

        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of %s (deg. C)' % (server_id))

        plt.savefig((f"{tmp_figures_folder}/{figure_label}.pdf")
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

    return rmse, rmse_perc


def unscale(x, scaler):
    return scaler.inverse_transform(np.array([np.append(np.copy(x), 0)]))[:, :-1]


def rescale(x, scaler):
    return scaler.transform(np.array([np.append(np.copy(x), 0)]))[:, :-1]


def remove_delta_temp(x, additional_variables, delta_temp_unscaled, scaler):
    unscaled_x = unscale(x, scaler)

    for idx, e in enumerate(additional_variables, start=x.shape[1] - len(additional_variables)):
        if "temp" in e.get("name"):
            unscaled_x[-1, idx] -= delta_temp_unscaled

    rescaled_x = rescale(unscaled_x, scaler)

    return rescaled_x


def evaluate_prediction_power(x, y, tss, server_id, learning_method, servers_names_raw, use_scaler, oracle_path=None, oracle=None, scaler=None, scaler_path=None, tmp_figures_folder=None, figure_label=""):

    server_idx = servers_names_raw.index(server_id)

    if oracle is None:
        if oracle_path is None:
            raise Exception("Please provide one of those arguments: ['oracle', 'oracle_path']")
        if learning_method == "neural":
            oracle = load_model(oracle_path)
        else:
            oracle = joblib.load(oracle_path)

    if scaler is None:
        if scaler_path is None:
            raise Exception("Please provide one of those arguments: ['scaler', 'scaler_path']")
        scaler = joblib.load(scaler_path)

    start_step = 0
    end_step = len(y)

    plot_data = []

    test_input_reusing_past_pred = None
    time_periods_without_real_temp = 20

    delta_temp_unscaled = None

    for (idx, (e, ts)) in enumerate(zip(y, tss)):

        ts_to_date = dt.datetime(*[int(x)
                                   for x in [a.split("-") + b.split(":")
                                             for (a,b) in [ts.replace("Z", "").split("T")]][0]])

        if idx < 0:
            continue

        if idx > 400:
            break

        if idx > 0 and (idx % time_periods_without_real_temp) != 0:
            resync = False
        else:
            resync = True

        test_input = np.array([x[idx]])
        if test_input_reusing_past_pred is None:
            test_input_reusing_past_pred = test_input

        expected_value = y[idx]

        if learning_method == "gaussian":
            result, sigma = oracle.predict(test_input)
        else:
            result = oracle.predict(test_input)

        test_input_copy = test_input.copy()

        if not resync:
            plop = remove_delta_temp(test_input_copy, ADDITIONAL_VARIABLES, delta_temp_unscaled, scaler)
            test_input_copy = plop

        if learning_method == "gaussian":
            result_reusing_past_pred, sigma_reusing_past_pred = oracle.predict(test_input_copy)
        else:
            result_reusing_past_pred = oracle.predict(test_input_copy)

        if use_scaler:
            unscaled_expected_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), expected_value)]))
            unscaled_predicted_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), result)]))
            unscaled_predicted_values_reusing_past_pred = scaler.inverse_transform(np.array([np.append(np.copy(test_input_copy), result_reusing_past_pred)]))

            expected_temp = unscaled_expected_values[:, -len(expected_value):][0]
            predicted_temp = unscaled_predicted_values[:, -len(expected_value):][0]
            predicted_temp_reusing_past_pred = unscaled_predicted_values_reusing_past_pred[:, -len(expected_value):][0]

            test_input_reusing_past_pred = test_input.copy()

            if idx > 0:
                unscaled_expected_values_copy = unscaled_expected_values.copy()
                unscaled_expected_values_copy[0][1] = unscaled_predicted_values_reusing_past_pred[0][-1]
                scaled_expected_values_copy = scaler.transform(unscaled_expected_values_copy)
                scaled_last_past_temperature = scaled_expected_values_copy[0][1]
            else:
                if learning_method == "gaussian":
                    scaled_last_past_temperature = result_reusing_past_pred[0]
                else:
                    scaled_last_past_temperature = result_reusing_past_pred[0][0]
            # print(f"   b:{scaler.transform(unscaled_predicted_values_reusing_past_pred)}")

            if resync:
                delta_temp_unscaled = expected_temp - predicted_temp_reusing_past_pred
        else:
            expected_temp = expected_value
            predicted_temp = result

        mse = ((predicted_temp_reusing_past_pred - expected_temp) ** 2)

        plot_data += [{
            "x": idx,
            "y_actual": expected_value,
            "y_pred": result_reusing_past_pred,
            # "temp_room": temp_room,
            "mse_mean": mse.mean(axis=0),
            "mse_raw": mse,
            "rmse_mean": np.sqrt(mse).mean(axis=0),
            "rmse_raw": np.sqrt(mse),
            "temp_actual": expected_temp,
            "temp_pred": predicted_temp,
            "temp_pred_reusing_past_pred": predicted_temp_reusing_past_pred,
            "resync": resync,
            "ts_to_date": ts_to_date
        }]

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 95)].mean()

    if tmp_figures_folder is not None:

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]
        y3_data = [d["temp_pred_reusing_past_pred"][server_idx] for d in sorted_plot_data]

        synced_sorted_dts = [d["ts_to_date"] for d in sorted_plot_data if d["resync"]]
        y4_data = [d["temp_pred_reusing_past_pred"][server_idx] for d in sorted_plot_data if d["resync"]]

        dts = [d["ts_to_date"] for d in sorted_plot_data]

        sorted_dts = sorted(dts)
        sorted_dts = sorted_dts[0:len(sorted_plot_data)]

        ax.plot(sorted_dts, y1_data, color='blue', label='actual temp.', linewidth=0.5)
        ax.plot(sorted_dts, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)
        ax.plot(sorted_dts, y3_data, color='green', label='predicted temp. (reusing past pred.)', alpha=0.5, linewidth=1.5)
        ax.scatter(synced_sorted_dts, y4_data, color='orange', label='sync', alpha=0.5, linewidth=0.5)
        plt.legend()

        ax2 = ax.twinx()
        ax2.legend(loc=0)
        plt.legend()

        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of %s (deg. C)' % (server_id))

        plt.savefig((f"{tmp_figures_folder}/{figure_label}_prediction_power.pdf")
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

    return rmse, rmse_perc
