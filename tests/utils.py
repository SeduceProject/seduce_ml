import pandas
import datetime
from seduce_ml.oracle.oracle import Oracle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from seduce_ml.data.data_from_api import DataResult

FAKE_VARIABLES = [
    {
        "name": f"x",
        "server_temperature": f"ecotype-1",
        "shift": True,
        "shift_count": 1,
    },
    {
        "name": f"y",
        "server_temperature": f"ecotype-1",
        "shift": True,
        "shift_count": 1,
    },
    {
        "name": f"z",
        "server_temperature": f"ecotype-1",
        "output": True,
        "become": f"y",
        "output_of": f"ecotype-1",
    }
]

SERVERS = ["ecotype-1"]

METADATA = {
    'input': ['x', 'y'],
    'output': ['z'],
    'variables': FAKE_VARIABLES,
    'input_variables': [var for var in FAKE_VARIABLES if not var.get("output", False)],
    'output_variables': [var for var in FAKE_VARIABLES if var.get("output", False)]
}


def generate_fake_data():
    now = datetime.datetime.now()

    metadata = METADATA

    input_columns_count = 2
    output_columns_count = 1

    inputs = []
    outputs = []
    timestamps_labels = []

    for i in range(0, 100):
        date = now + datetime.timedelta(minutes=1)
        timestamp_label = date
        x = float(i)
        y = float(i) + 1
        z = x + y
        inputs += [[x, y]]
        outputs += [z]
        timestamps_labels += [timestamp_label]

    inputs = np.array(inputs).reshape(100, 2)
    outputs = np.array(outputs).reshape(100, 1)

    scaler = None

    # Scale values
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    all_non_scaled_values = np.copy(inputs)
    all_non_scaled_values = np.append(all_non_scaled_values, outputs, axis=1)
    scaled_values = scaler.fit_transform(all_non_scaled_values)

    unscaled_x, unscaled_y = all_non_scaled_values[:, :input_columns_count], all_non_scaled_values[:, -output_columns_count:]
    scaled_x, scaled_y = scaled_values[:, :input_columns_count], scaled_values[:, -output_columns_count:]

    # Create dataframes with all data
    complete_data_scaled = pandas.concat([pandas.DataFrame(scaled_x, columns=metadata.get("input")),
                                          pandas.DataFrame(scaled_y, columns=metadata.get("output")),
                                          pandas.DataFrame(timestamps_labels, columns=["timestamp"])],
                                         axis=1)
    complete_data_unscaled = pandas.concat([pandas.DataFrame(unscaled_x, columns=metadata.get("input")),
                                            pandas.DataFrame(unscaled_y, columns=metadata.get("output")),
                                            pandas.DataFrame(timestamps_labels, columns=["timestamp"])],
                                           axis=1)

    # Export the complete dataframe to csv
    complete_data_scaled.to_csv("tests_data/complete_data_scaled.csv")
    complete_data_unscaled.to_csv("tests_data/complete_data_unscaled.csv")

    return DataResult(SERVERS, metadata, (100, 2), (100, 1), scaler, "tests_data/complete_data_scaled.csv", "tests_data/complete_data_unscaled.csv")


class FakeOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def predict(self, unscaled_input_values):
        return np.sum(unscaled_input_values).reshape(1, 1) + 0.1
