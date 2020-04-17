import pandas
import datetime
from seduce_ml.oracle.oracle import Oracle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

    selected_servers_names_raw = SERVERS

    metadata = METADATA

    input_columns_count = 2
    output_columns_count = 1

    inputs = []
    outputs = []
    timestamps_labels = []

    for i in range(0, 100):
        date = now + datetime.timedelta(minutes=1)
        timestamp_label = date.strftime("%Y-%s-%dT")
        x = i
        y = i + 1
        z = x + y
        inputs += [[x, y]]
        outputs += [z]
        timestamps_labels += [timestamp_label]

    inputs = np.array(inputs).reshape(100, 2)
    outputs = np.array(outputs).reshape(100, 1)

    comon_result_properties = {
        "unscaled_x": inputs,
        "unscaled_y": outputs,
        "unscaled_x_df": pandas.DataFrame(inputs, columns=metadata.get("input")),
        "unscaled_y_df": pandas.DataFrame(outputs, columns=metadata.get("output")),
        "timestamps_labels": timestamps_labels,
        "tss": timestamps_labels,
        "data": {},
        "shape": (2, 1),
        "selected_servers_names_raw": selected_servers_names_raw,
        "servers_names_raw": selected_servers_names_raw,
        "metadata": metadata
    }

    scaler = None

    # Scale values
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    all_non_scaled_values = np.copy(inputs)
    all_non_scaled_values = np.append(all_non_scaled_values, outputs, axis=1)
    scaled_values = scaler.fit_transform(all_non_scaled_values)

    scaled_x, scaled_y = scaled_values[:, :input_columns_count], scaled_values[:, -output_columns_count:]

    additional_result_properties = {
        "x": scaled_x,
        "y": scaled_y,
        "x_df": pandas.DataFrame(scaled_x, columns=metadata.get("input")),
        "y_df": pandas.DataFrame(scaled_y, columns=metadata.get("output")),
        "scaler": scaler
    }

    result = {**comon_result_properties, **additional_result_properties}

    return result


class FakeOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def predict(self, unscaled_input_values):
        return np.sum(unscaled_input_values).reshape(1, 1)
