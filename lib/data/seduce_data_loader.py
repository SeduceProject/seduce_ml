import numpy as np
import requests
from functional import seq
import os
import json
import calendar
import time
from sklearn.preprocessing import MinMaxScaler


def compute_average_consumption(x):
    return np.mean(x)


def generate_real_consumption_data(start_date=None,
                                   end_date=None,
                                   show_progress=True,
                                   data_file_path=None,
                                   group_by=60,
                                   scaler=None,
                                   use_scaler=True,
                                   additional_variables=None):

    if data_file_path is None:
        data_file_path = "data/data_60m.json"

    if additional_variables is None:
        additional_variables = []

    seduce_infrastructure_tree = requests.get("https://api.seduce.fr/infrastructure/description/tree").json()

    servers_names_raw = [f"ecotype-{i}" for i in range(1, 49)]
    servers_names_raw = sorted(servers_names_raw, key=lambda s: int(s.split("-")[1]))

    # selected_servers_names_raw = [f"ecotype-{i}" for i in range(37, 49)]
    selected_servers_names_raw = [f"ecotype-{i}" for i in range(40, 41)]
    selected_servers_names_raw = sorted(selected_servers_names_raw, key=lambda s: int(s.split("-")[1]))

    servers_names = seq(servers_names_raw)
    selected_servers_names = seq(selected_servers_names_raw)

    reload_data = False
    # reload_data = True
    if not os.path.exists(data_file_path):
        reload_data = True
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)
            if data.get("start_date") != start_date or data.get("end_date") != end_date or data.get("group_by") != group_by:
                reload_data = True

    if reload_data:

        data = {
            "consumptions": {},
            "start_date": start_date,
            "end_date": end_date,
            "group_by": group_by,
            "room_temperature": [],
            "return_air_temperature": [],
            "min_consumption": None,
            "max_consumption": None,
            "min_temperature": None,
            "max_temperature": None,
            "server_names": servers_names_raw,
            "selected_servers": selected_servers_names_raw
        }

        epoch_times = [calendar.timegm(time.strptime(t, '%Y-%m-%dT%H:%M:%S.000Z')) for t in [start_date, end_date]]
        start_epoch = min(epoch_times)
        end_epoch = max(epoch_times)

        group_by_seconds = "%ss" % (group_by * 60)
        dump_data_url = """https://dashboard.seduce.fr/dump/all/aggregated?start_date=%ss&end_date=%ss&group_by=%s""" % (start_epoch, end_epoch, group_by_seconds)
        dump_data = requests.get(dump_data_url).json()

        incomplete_series = [v
                             for v in dump_data["sensors_data"].values()
                             if len(v.get("timestamps", [])) != max([len(v1.get("timestamps"))
                                                                     for v1 in dump_data["sensors_data"].values()
                                                                     ])
                             ]

        if len(incomplete_series) > 0:
            raise Exception("Some series are incomplete!")

        data["timestamps"] = list(set([ts
                                       for v in dump_data["sensors_data"].values()
                                       for ts in v.get("timestamps")
                                       ]))

        data["sensors_data"] = dump_data["sensors_data"]

        # get consumptions of servers
        for server_name in servers_names:

            # find PDUS
            node_pdus = seduce_infrastructure_tree["power"][server_name].values()

            # find back_temperature_sensors
            back_temperature_sensor = [sensor
                                       for x in seduce_infrastructure_tree["temperature"].values()
                                       for (side, sensor_bus) in x.items()
                                       if side == "back"
                                       for sensor in sensor_bus.values()
                                       if server_name in sensor.get("tags", [])
                                       ][0].get("serie")

            # Create a dict with all the infos related to the node
            data["consumptions"][server_name] = {
                "means": [sum(tuple_n) if None not in tuple_n else -1
                                       for tuple_n in zip(*[v.get("means")
                                                            for k,v in dump_data.get("sensors_data").items()
                                                            if k in node_pdus
                                                            ])
                                       ],
                # "temperatures": dump_data.get("sensors_data")[back_temperature_sensor]["maxs"],
                "temperatures": dump_data.get("sensors_data")[back_temperature_sensor]["means"],
                "timestamps": dump_data.get("sensors_data")[back_temperature_sensor]["timestamps"],
            }

            if show_progress:
                print('.', end='')

        # Detect incomplete data
        filter_timestamps = []
        for server_name in servers_names:
            data_server = data["consumptions"][server_name]

            ziped_big_array = zip(data_server["timestamps"], data_server["means"], data_server["temperatures"])
            filtered_ziped_big_array = [tuple_n
                                        for tuple_n in ziped_big_array
                                        if -1 in [tuple_n[1], tuple_n[2]] or None in [tuple_n[1], tuple_n[2]]
                                        ]
            if len(filtered_ziped_big_array) > 0:
                filter_timestamps += [x[0] for x in filtered_ziped_big_array]
        data["filter_timestamps"] = list(set(filter_timestamps))

        # Filter incomplete data
        for server_name in servers_names:
            data_server = data["consumptions"][server_name]

            ziped_big_array = zip(data_server["timestamps"], data_server["means"], data_server["temperatures"])
            filtered_ziped_big_array = [tuple_n
                                        for tuple_n in ziped_big_array
                                        if tuple_n[0] not in filter_timestamps
                                        ]

            data_server["timestamps"] = [tuple_n[0] for tuple_n in filtered_ziped_big_array]
            data_server["means"] = [tuple_n[1] for tuple_n in filtered_ziped_big_array]
            data_server["temperatures"] = [tuple_n[2] for tuple_n in filtered_ziped_big_array]

        # Normalize data
        for server_name in servers_names:
            data_server = data["consumptions"][server_name]

            data_server["means"] = [x
                                    for x in data_server["means"]]
            data_server["temperatures"] = [x
                                           for x in data_server["temperatures"]]

        for key in ["room_temperature", "return_air_temperature"]:
            data[key] = [tuple_2[1]
                         for tuple_2 in zip(data["timestamps"], data[key])
                         if tuple_2[0] not in filter_timestamps]

        with open(data_file_path, "w+") as data_file:
            json.dump(data, data_file)
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)

    for additional_var in additional_variables:
        if "sensor" in additional_var:
            additional_var_dict = data.get("sensors_data")[additional_var.get("sensor")]
            variables_values = [b
                                for (a, b) in zip(additional_var_dict["timestamps"],
                                                  additional_var_dict["means"])
                                if a not in data.get("filter_timestamps")]
        elif "server_consumption" in additional_var:
            additional_var_dict = data.get("consumptions")[additional_var.get("server_consumption")]
            variables_values = [b
                                for (a, b) in zip(additional_var_dict["timestamps"],
                                                  additional_var_dict["means"])
                                if a not in data.get("filter_timestamps")]
        elif "server_temperature" in additional_var:
            additional_var_dict = data.get("consumptions")[additional_var.get("server_temperature")]
            variables_values = [b
                                for (a, b) in zip(additional_var_dict["timestamps"],
                                                  additional_var_dict["temperatures"])
                                if a not in data.get("filter_timestamps")]
        elif "consumption_function" in additional_var:
            all_data = data.get("consumptions")
            func = additional_var["consumption_function"]

            consumptions = {}
            for server_name in all_data.keys():
                consumptions[server_name] = [
                    cons
                    for (ts, cons) in zip(all_data[server_name]["timestamps"], all_data[server_name]["means"])
                    if ts not in data["filter_timestamps"]
                ]

            variables_values = [func(i) for i in zip(*consumptions.values())]
        else:
            raise Exception(f"Could not understand how to compute the additional variable {additional_var.get('name', str(additional_var))}")

        if not additional_var.get("shift", False):
            data[additional_var.get("name")] = variables_values
        else:
            data[additional_var.get("name")] = [variables_values[0]] + variables_values[0:-1]

    x = None
    y = None

    timestamps_labels = None

    for server_name in servers_names_raw:
        if timestamps_labels is not None:
            if data["consumptions"][server_name]["timestamps"] != timestamps_labels:
                raise Exception(f"Timestamps of server {server_name} don't match timestamps of other servers")
        timestamps_labels = data["consumptions"][server_name]["timestamps"]

    for selected_server in selected_servers_names:

        if timestamps_labels is not None:
            if data["consumptions"][selected_server]["timestamps"] != timestamps_labels:
                print("plop")

        timestamps_labels = data["consumptions"][selected_server]["timestamps"]
        consumption_values = data["consumptions"][selected_server]["means"]
        temperature_values = data["consumptions"][selected_server]["temperatures"]

        z = np.array(seq(consumption_values).map(lambda x: x if x is not None else 0).to_list()).reshape(len(timestamps_labels), 1)
        # Add values of the additional variable to the 'x' array
        if x is None:
            x = z
        else:
            x = np.append(x, z, axis=1)
        z = np.array(seq(temperature_values).map(lambda x: x if x is not None else 0).to_list()).reshape(len(timestamps_labels), 1)
        # Add values of the additional variable to the 'x' array
        if y is None:
            y = z
        else:
            y = np.append(y, z, axis=1)

    for additional_var in additional_variables:

        if additional_var.get("exclude_from_training_data", False):
            continue

        temperatures = data[additional_var.get("name")]
        z = np.array(seq(temperatures).map(lambda x: x if x is not None else 0).to_list()).reshape(len(x), 1)
        # Add values of the additional variable to the 'x' array
        x = np.append(x, z, axis=1)

    input_columns_count = x.shape[1]
    output_columns_count = y.shape[1]

    shape = (input_columns_count, output_columns_count)

    if use_scaler:
        # Scale values
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))

        all_non_scaled_values = np.copy(x)
        all_non_scaled_values = np.append(all_non_scaled_values, y, axis=1)
        scaled_values = scaler.fit_transform(all_non_scaled_values)

        scaled_x, scaled_y = scaled_values[:, :input_columns_count], scaled_values[:, -output_columns_count:]

        return scaled_x, scaled_y, timestamps_labels, data, scaler, shape, selected_servers_names_raw
    else:
        return x, y, timestamps_labels, data, None, shape, selected_servers_names_raw
