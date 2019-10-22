from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import datetime as dt
from lib.data.seduce_data_loader import get_additional_variables


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


def unscale_input(x, scaler):
    scaler_columns_count = scaler.scale_.shape[0]
    input_columns_count = x.shape[1]
    temp_array = np.append(x, np.zeros((scaler_columns_count - input_columns_count)))
    return scaler.inverse_transform(np.array([temp_array]))[:, :input_columns_count]


def rescale_input(x, scaler):
    scaler_columns_count = scaler.scale_.shape[0]
    input_columns_count = x.shape[1]
    temp_array = np.append(x, np.zeros((scaler_columns_count - input_columns_count)))
    return scaler.transform(np.array([temp_array]))[:, :input_columns_count]


def unscale_output(y, scaler):
    scaler_columns_count = scaler.scale_.shape[0]
    output_columns_count = y.shape[0]
    temp_array = np.append(np.zeros((scaler_columns_count - output_columns_count)), y)
    return scaler.inverse_transform(np.array([temp_array]))[:, -output_columns_count:]


def rescale_output(y, scaler):
    scaler_columns_count = scaler.scale_.shape[0]
    output_columns_count = y.shape[0]
    temp_array = np.append(np.zeros((scaler_columns_count - output_columns_count)), y)
    return scaler.inverse_transform(np.array([temp_array]))[:, -output_columns_count:]


def shift_data(x, additional_variables, scaler, predicted_temp_reusing_past_pred):
    unscaled_x = unscale_input(x, scaler)

    unscaled_x_old = unscaled_x.copy()

    shifted_old_temp_values_map = {}

    # Extract value in order to easily shift them (n-1 => n-2; n-2 => n-3; ...)
    for idx, e in enumerate(additional_variables, start=x.shape[1] - len(additional_variables)):
        if all([pattern in e.get("name") for pattern in ["temp", "past_temp", "n-"]]):
            hostname, time_step_str = e.get("name").split("_past_temp_n-")
            time_step = int(time_step_str)
            shifted_old_temp_values_map[f"{hostname}_{time_step}"] = unscaled_x_old[-1, idx]

    for idx, e in enumerate(additional_variables, start=x.shape[1] - len(additional_variables)):
        # Reuse the past prediction
        if all([pattern in e.get("name") for pattern in ["temp", "past_temp", "n-"]]):
            hostname, time_step_str = e.get("name").split("_past_temp_n-")
            time_step = int(time_step_str)

            if time_step == 1:
                unscaled_x[-1, idx] = predicted_temp_reusing_past_pred
            else:
                unscaled_x[-1, idx] = shifted_old_temp_values_map[f"{hostname}_{time_step-1}"]

    # print(f"{unscaled_x_old} vs {unscaled_x}")

    rescaled_x = rescale_input(unscaled_x, scaler)

    return rescaled_x


def evaluate_prediction_power(x, y, tss, server_id, learning_method, servers_names_raw, use_scaler, oracle_path=None, oracle=None, scaler=None, scaler_path=None, tmp_figures_folder=None, figure_label="", produce_figure=True):

    additional_variable = get_additional_variables(server_id)

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
    time_periods_without_real_temp = 10

    resync = True
    resync_count = 0
    expected_temp = None

    for (idx, (e, ts)) in enumerate(zip(y, tss)):

        ts_to_date = dt.datetime(*[int(x)
                                   for x in [a.split("-") + b.split(":")
                                             for (a, b) in [ts.replace("Z", "").split("T")]][0]])

        # if idx < 0:
        #     continue
        #
        # if idx > 800:
        #     break

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
            test_input_copy = shift_data(test_input_copy, additional_variable, scaler, predicted_temp_reusing_past_pred)
        else:
            if expected_temp is not None:
                test_input_copy = shift_data(test_input_copy, additional_variable, scaler, expected_temp)

        if learning_method == "gaussian":
            result_reusing_past_pred, sigma_reusing_past_pred = oracle.predict(test_input_copy)
        else:
            result_reusing_past_pred = oracle.predict(test_input_copy)

        if type(result) == list:
            result = result[0]
        if type(result_reusing_past_pred) == list:
            result_reusing_past_pred = result_reusing_past_pred[0]

        expected_temp = unscale_output(expected_value, scaler)[0]
        predicted_temp = unscale_output(result, scaler)[0]

        if not resync:
            predicted_temp_reusing_past_pred = unscale_output(result_reusing_past_pred, scaler)[0]
        else:
            predicted_temp_reusing_past_pred = predicted_temp

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
            # "temp_pred_reusing_past_pred2": predicted_temp_reusing_past_pred2,
            "resync": resync,
            "ts_to_date": ts_to_date
        }]


    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 95)].mean()

    if tmp_figures_folder is not None and produce_figure:

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]
        y3_data = [d["temp_pred_reusing_past_pred"][server_idx] for d in sorted_plot_data]
        # y5_data = [d["temp_pred_reusing_past_pred2"][server_idx] for d in sorted_plot_data]

        synced_sorted_dts = [d["ts_to_date"] for d in sorted_plot_data if d["resync"]]
        y4_data = [d["temp_pred_reusing_past_pred"][server_idx] for d in sorted_plot_data if d["resync"]]

        dts = [d["ts_to_date"] for d in sorted_plot_data]

        sorted_dts = sorted(dts)
        sorted_dts = sorted_dts[0:len(sorted_plot_data)]

        ax.plot(sorted_dts, y1_data, color='blue', label='actual temp.', linewidth=0.5)
        ax.plot(sorted_dts, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)
        ax.plot(sorted_dts, y3_data, color='green', label='predicted temp. (reusing past pred.)', alpha=0.5, linewidth=0.8)
        # ax.plot(sorted_dts, y5_data, color='black', label='predicted temp. (reusing past pred.)', alpha=0.5, linewidth=0.8)
        ax.scatter(synced_sorted_dts, y4_data, color='orange', marker='x', label='sync', alpha=0.5, linewidth=0.5)

        for i, (a, b) in enumerate(zip(synced_sorted_dts, y4_data)):
            ax.annotate(f"{i+1}", (a, b), fontsize=5)

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
