import sys
from keras.models import load_model
from lib.data.seduce_data_loader import generate_real_consumption_data
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# NETWORK_PATH = 'data/seduceml_2019_07_17_T_14_37_00.h5'
# NETWORK_PATH = 'data/seduceml_2019_07_18_T_15_48_18.h5'
# NETWORK_PATH = 'data/seduceml_2019_07_18_T_15_00_12.h5'
# NETWORK_PATH = 'data/seduceml_2019_07_18_T_16_04_23.h5'
# NETWORK_PATH = 'data/seduceml_2019_07_18_T_16_30_59.h5'
NETWORK_PATH = "last"

start_date = "2019-07-11T01:00:00.000Z"
end_date = "2019-07-14T00:00:00.000Z"


if __name__ == "__main__":

    if NETWORK_PATH == "last":
        netwok_path = "data/%s" % (sorted(os.listdir("data"))[-1])
    else:
        netwok_path = NETWORK_PATH

    oracle = load_model(netwok_path)

    with open("data.json", "r") as data_file:
        training_data = json.load(data_file)

    x, y, tss, data = generate_real_consumption_data(start_date, end_date, data_file_path="data_validation.json", training_data=training_data)

    start_step = 0
    end_step = len(y)

    plot_data = []

    for idx in range(0, len(y)):
        test_input = np.array([x[idx]])
        expected_value = y[idx]
        result = oracle.predict(test_input)[0]

        plot_data += [{
            "x": idx,
            "y_actual": expected_value,
            "y_pred": result,
            "temp_room": data.get("room_temperature")[idx]
        }]

    sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

    fig = plt.figure()
    ax = plt.axes()

    server_id = 32
    # server_id = 2

    x_data = [d["x"] for d in sorted_plot_data]
    y1_data = [d["y_actual"][server_id] * (training_data["max_temperature"] - training_data["min_temperature"]) + training_data["min_temperature"]
               for d in sorted_plot_data]
    y2_data = [d["y_pred"][server_id] * (training_data["max_temperature"] - training_data["min_temperature"]) + training_data["min_temperature"]
               for d in sorted_plot_data]
    y3_data = [d["temp_room"]
               for d in sorted_plot_data]
    x_data = range(0, len(y1_data))

    ax.plot(x_data, y1_data, color='blue', label='actual temp.')
    ax.plot(x_data, y2_data, color='red', label='predicted temp.', alpha=0.5)
    # ax.plot(x_data, y3_data, color='green', label='room temp.', alpha=0.5)

    plt.legend()

    plt.xlabel('Time (hour)')
    plt.ylabel('Back temperature of ecotype-%s (deg. C)' % (server_id+1))

    plt.show()

    sys.exit(0)
