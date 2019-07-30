from keras.models import load_model
from lib.data.seduce_data_loader import generate_real_consumption_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib


def validate_seduce_ml(start_date, end_date, group_by, comparison_plot_data, server_id):

    for PLOT_DATA in comparison_plot_data:

        oracle = load_model(PLOT_DATA["dump_path"])

        scaler = joblib.load(PLOT_DATA["scaler_path"])

        x, y, tss, data, scaler, shape =\
            generate_real_consumption_data(start_date,
                                           end_date,
                                           data_file_path="data_validation.json",
                                           group_by=group_by,
                                           scaler=scaler)

        start_step = 0
        end_step = len(y)

        plot_data = []

        for idx in range(0, len(y)):
            test_input = np.array([x[idx]])
            expected_value = y[idx]
            result = oracle.predict(test_input)[0]

            unscaled_expected_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), expected_value)]))
            unscaled_predicted_values = scaler.inverse_transform(np.array([np.append(np.copy(test_input), result)]))

            expected_temp = unscaled_expected_values[:, -len(expected_value):][0]
            predicted_temp = unscaled_predicted_values[:, -len(expected_value):][0]

            mse = ((predicted_temp - expected_temp) ** 2)

            neighbours = [i for i in range(0, 12) if i != server_id]

            server_power_consumption_neighbours = -1
            if y.shape[1] > len(neighbours):
                server_power_consumption_neighbours = np.mean(unscaled_predicted_values[0][neighbours])

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
                "server_power_consumption": unscaled_predicted_values[0][server_id],
                "server_power_consumption_neighbours": server_power_consumption_neighbours
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

        # date_str = time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime(epoch))
        # neural_net_dump_path = "data/seduceml_%s.h5" % date_str

        PLOT_DATA["mse"] = mse
        PLOT_DATA["mse_perc"] = mse_perc
        PLOT_DATA["rmse"] = rmse
        PLOT_DATA["rmse_perc"] = rmse_perc

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        x_data = [d["x"] for d in sorted_plot_data]
        y1_data = [d["temp_actual"][server_id] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][server_id] for d in sorted_plot_data]
        y3_data = [d["temp_room"] for d in sorted_plot_data]
        y4_data = [d["server_power_consumption"] for d in sorted_plot_data]
        y5_data = [d["server_power_consumption_neighbours"] for d in sorted_plot_data]
        x_data = range(0, len(y1_data))

        ax.plot(x_data, y1_data, color='blue', label='actual temp.')
        ax.plot(x_data, y2_data, color='red', label='predicted temp.', alpha=0.5)
        # ax.plot(x_data, y3_data, color='green', label='room temp.', alpha=0.5)

        #ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        #ax2.plot(x_data, y4_data, color='green', label='power consumption.', alpha=0.5)
        #ax2.plot(x_data, y5_data, color='orange', label='power consumption. neighbours', alpha=0.5)

        plt.legend()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of ecotype-%s (deg. C)' % (server_id+1))

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
