from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import datetime as dt


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

    for idx, e in enumerate(y):
        test_input = np.array([x[idx]])
        expected_value = y[idx]

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

        plt.savefig((f"{tmp_figures_folder}/{figure_label}.pdf")
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

    return rmse, rmse_perc
