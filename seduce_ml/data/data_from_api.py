import numpy as np
import requests
from functional import seq
import os
import json
import calendar
import time
from sklearn.preprocessing import MinMaxScaler
import pandas
from sklearn.model_selection import train_test_split

IGNORE_PATTERNS = ["_I_AC", "_P_AC", "b232", "net_", "apparent", "reactive", "battery", "active", "weather"]


def get_additional_variables(server_id, learning_method):
    _, server_num_str = server_id.split("-")
    server_num = int(server_num_str)

    variables = [
        {
            "name": f"aggregate_cluster_consumption",
            "sensor": f"hardware_cluster",
            "aggregate": True,
        },
    ]

    variables += [{
        "name": f"ecotype_{i}_consumption",
        "server_consumption": f"ecotype-{i}",
    }
        for i in range(37, 49)]

    output_range = [server_num]

    variables += [{
        "name": f"ecotype_{i}_temperature_past_1",
        "server_temperature": f"ecotype-{i}",
        "shift": True,
        "shift_count": 1,
    }
        for i in output_range
    ]

    variables += [{
        "name": f"ecotype_{i}_temperature",
        "server_temperature": f"ecotype-{i}",
        "output": True,
        "become": f"ecotype_{i}_temperature_past_1",
        "output_of": f"ecotype-{i}",
    }
        for i in output_range
    ]

    return variables


def get_additional_variables_multiple_servers(server_ids, learning_method):
    result = []

    for server_id in server_ids:
        result += get_additional_variables(server_id, learning_method)

    return result


def generate_real_consumption_data(start_date=None,
                                   end_date=None,
                                   show_progress=True,
                                   data_file_name=None,
                                   data_folder_path=None,
                                   group_by=60,
                                   scaler=None,
                                   use_scaler=True,
                                   server_id=None,
                                   server_ids=None,
                                   learning_method=None):

    if data_folder_path is None:
        raise Exception("Please specify 'data_folder_path' parameter")

    if data_file_name is None:
        data_file_name = f"data_{ group_by }m.json"

    data_file_path = f"{ data_folder_path }/{ data_file_name }"

    seduce_infrastructure_tree = requests.get("https://api.seduce.fr/infrastructure/description/tree").json()

    servers_names_raw = [f"ecotype-{i}" for i in range(1, 49)]
    servers_names_raw = sorted(servers_names_raw, key=lambda s: int(s.split("-")[1]))

    if server_id is not None and server_ids is None:
        selected_servers_names_raw = [server_id]
    elif server_id is None and server_ids is not None:
        selected_servers_names_raw = server_ids
    else:
        raise Exception("Could not figure which server to use")

    raw_variables = get_additional_variables_multiple_servers(selected_servers_names_raw, learning_method)

    # Remove duplicate variables
    variables = [v for (k, v) in dict([(f"{var}", var) for var in raw_variables]).items()]

    # Create explicit link between linked variables
    for var in variables:
        if "become" in var:
            [becoming_variable] = [v
                                   for v in variables
                                   if v.get("name") == var.get("become")]
            if becoming_variable is not None:
                becoming_variable["based_on"] = var.get("name")

    selected_servers_names_raw = sorted(selected_servers_names_raw, key=lambda s: int(s.split("-")[1]))

    servers_names = seq(servers_names_raw)

    reload_data = False
    # reload_data = True
    if not os.path.exists(data_file_path):
        reload_data = True
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)
            if data.get("start_date") != start_date or data.get("end_date") != end_date or data.get(
                    "group_by") != group_by:
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

        dump_data = {}
        print("Fetching data from API:")
        for i in range(start_epoch, end_epoch, 24 * 3600):
            subrequest_start = i
            subrequest_end = min(subrequest_start + 24 * 3600, end_epoch)

            _dump_data_url = f"""https://dashboard.seduce.fr/dump/all/aggregated?start_date={subrequest_start}s&end_date={subrequest_end}s&group_by={group_by_seconds}"""

            try:
                _dump_data = requests.get(_dump_data_url).json()
            except Exception as e:
                # print(f"There was an error fetching data at this url: '{_dump_data_url}'. I will continue")
                print('x', end='')
                continue

            if "start_date" not in dump_data:
                dump_data["start_date"] = _dump_data["start_date"]
                dump_data["is_downsampled"] = _dump_data["is_downsampled"]
                dump_data["timestamps"] = []
                dump_data["means"] = []
                dump_data["sensors_data"] = {}
            dump_data["end_date"] = _dump_data["end_date"]

            timestamps_counts = []
            for sensor_name, sensor_data in _dump_data["sensors_data"].items():
                timestamps_counts += [len(_dump_data["sensors_data"][sensor_name]["timestamps"])]

            if len(set(timestamps_counts)) > 1:
                print('x', end='')
                continue

            for sensor_name, sensor_data in _dump_data["sensors_data"].items():

                ignore_sensor = False
                for ignore_pattern in IGNORE_PATTERNS:
                    if ignore_pattern in sensor_name:
                        # print(f"ignore {sensor_name}")
                        ignore_sensor = True

                if ignore_sensor:
                    continue

                if sensor_name not in dump_data["sensors_data"]:
                    dump_data["sensors_data"][sensor_name] = {
                        "means": [],
                        "timestamps": [],
                        "maxs": []
                    }

                valid_t3 = [(ts, means, maxs)
                            for (ts, means, maxs) in zip(sensor_data.get("timestamps"),
                                                         sensor_data.get("means"),
                                                         sensor_data.get("maxs"))
                            if ts not in dump_data["sensors_data"][sensor_name]["timestamps"]]

                dump_data["sensors_data"][sensor_name]["timestamps"] += [ts for (ts, means, maxs) in valid_t3]
                dump_data["sensors_data"][sensor_name]["means"] += [means for (ts, means, maxs) in valid_t3]
                dump_data["sensors_data"][sensor_name]["maxs"] += [maxs for (ts, means, maxs) in valid_t3]

                # for item_key, item_value in sensor_data.items():
                #     if item_key not in dump_data["sensors_data"][sensor_name]:
                #         dump_data["sensors_data"][sensor_name][item_key] = []
                #     dump_data["sensors_data"][sensor_name][item_key] += item_value

            if show_progress:
                print('.', end='')

        incomplete_series = [v
                             for v in dump_data["sensors_data"].values()
                             if len(v.get("timestamps", [])) != max([len(v1.get("timestamps"))
                                                                     for v1 in dump_data["sensors_data"].values()
                                                                     ])
                             ]

        if len(incomplete_series) > 0:
            all_timestamps = []
            for v in dump_data["sensors_data"].values():
                all_timestamps += v.get("timestamps")
            all_timestamps = set(all_timestamps)

            problematic_sensor = {}
            problematic_timestamps = []
            incommon_timestamps = all_timestamps
            for k, v in dump_data["sensors_data"].items():
                missing_timestamps = [ts for ts in all_timestamps if ts not in v.get("timestamps")]
                incommon_timestamps = [ts for ts in incommon_timestamps if ts in v.get("timestamps")]

                if len(missing_timestamps) > 0:
                    problematic_sensor[k] = missing_timestamps
                    problematic_timestamps += missing_timestamps
            problematic_timestamps = set(problematic_timestamps)

            # Fix problematic timestamps
            for k, v in dump_data["sensors_data"].items():
                valid_t3 = [(ts, means, maxs)
                            for (ts, means, maxs) in zip(v.get("timestamps"),
                                                         v.get("means"),
                                                         v.get("maxs"))
                            if ts in incommon_timestamps]
                dump_data["sensors_data"][k]["timestamps"] = [ts for ts, means, maxs in valid_t3]
                dump_data["sensors_data"][k]["means"] = [means for ts, means, maxs in valid_t3]
                dump_data["sensors_data"][k]["maxs"] = [maxs for ts, means, maxs in valid_t3]
                print(f"{k} => {len([maxs for ts, means, maxs in valid_t3])}")

            # raise Exception("Some series are incomplete!")

        incomplete_series = [k
                             for k, v in dump_data["sensors_data"].items()
                             if len(v.get("timestamps", [])) != max([len(v1.get("timestamps"))
                                                                     for v1 in dump_data["sensors_data"].values()
                                                                     ])
                             ]

        data["timestamps"] = list(set([ts
                                       for v in dump_data["sensors_data"].values()
                                       for ts in v.get("timestamps")
                                       ]))

        data["sensors_data"] = dump_data["sensors_data"]

        # get consumptions of servers
        print("Consolidating consumptions of servers:")
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
                                               for k, v in dump_data.get("sensors_data").items()
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

        if not os.path.exists(data_file_path):
            permission_mode = "w"
        else:
            permission_mode ="w+"
        with open(data_file_path, permission_mode) as data_file:
            json.dump(data, data_file)
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)

    for var in variables:
        if "sensor" in var:
            if var.get("sensor") == "hardware_cluster":
                all_data = data.get("consumptions")

                consumptions = {}
                for server_name in all_data.keys():
                    consumptions[server_name] = [
                        cons
                        for (ts, cons) in zip(all_data[server_name]["timestamps"], all_data[server_name]["means"])
                        if ts not in data["filter_timestamps"]
                    ]

                variables_values = np.sum(np.array([v for k, v in consumptions.items()]), axis=0).tolist()

            else:
                additional_var_dict = data.get("sensors_data")[var.get("sensor")]
                variables_values = [b
                                    for (a, b) in zip(additional_var_dict["timestamps"],
                                                      additional_var_dict["means"])
                                    if a not in data.get("filter_timestamps")]
        elif "server_consumption" in var:
            additional_var_dict = data.get("consumptions")[var.get("server_consumption")]
            variables_values = [b
                                for (a, b) in zip(additional_var_dict["timestamps"],
                                                  additional_var_dict["means"])
                                if a not in data.get("filter_timestamps")]
        elif "server_temperature" in var:
            additional_var_dict = data.get("consumptions")[var.get("server_temperature")]
            variables_values = [b
                                for (a, b) in zip(additional_var_dict["timestamps"],
                                                  additional_var_dict["temperatures"])
                                if a not in data.get("filter_timestamps")]
        elif "consumption_function" in var:
            all_data = data.get("consumptions")
            func = var["consumption_function"]

            consumptions = {}
            for server_name in all_data.keys():
                consumptions[server_name] = [
                    cons
                    for (ts, cons) in zip(all_data[server_name]["timestamps"], all_data[server_name]["means"])
                    if ts not in data["filter_timestamps"]
                ]

            variables_values = [func(i) for i in zip(*consumptions.values())]
        else:
            raise Exception(f"Could not understand how to compute the additional variable {var.get('name', str(var))}")

        if not var.get("shift", False):
            data[var.get("name")] = variables_values
        else:
            shift_count = var.get("shift_count", 1)
            new_values = [variables_values[shift_count - 1] for x in range(0, shift_count)] + variables_values[
                                                                                              0:-shift_count]
            data[var.get("name")] = new_values

        if var.get("difference", False):
            differences = [current - prev
                           for (prev, current) in zip(data[var.get("name")][0:1] + data[var.get("name")][0:-1],
                                                      data[var.get("name")]
                                                      )]
            data[var.get("name")] = differences

        if "rescale" in var:
            data[var.get("name")] = [var.get("rescale")(x) for x in data[var.get("name")]]

    x = None
    y = None

    timestamps_labels = None

    for server_name in servers_names_raw:
        if timestamps_labels is not None:
            if data["consumptions"][server_name]["timestamps"] != timestamps_labels:
                raise Exception(f"Timestamps of server {server_name} don't match timestamps of other servers")
        timestamps_labels = data["consumptions"][server_name]["timestamps"]

    timestamps = [pandas.Timestamp(ts_label)
                  for ts_label in timestamps_labels]

    metadata = {
        "input": [],
        "output": [],
        "variables": [],
        "input_variables": [],
        "output_variables": [],
    }

    for var in variables:

        if var.get("exclude_from_training_data", False):
            continue

        temperatures = data[var.get("name")]

        z = np.array(seq(temperatures).map(lambda x: x if x is not None else 0).to_list()).reshape(
            len(timestamps_labels), 1)
        # Add values of the additional variable to the 'x' array

        if not var.get("output", False):
            if x is None:
                x = z
            else:
                x = np.append(x, z, axis=1)
            metadata["input"] += [var.get("name")]
            metadata["input_variables"] += [var]
        else:
            if y is None:
                y = z
            else:
                y = np.append(y, z, axis=1)
            metadata["output"] += [var.get("name")]
            metadata["output_variables"] += [var]
        metadata["variables"] += [var]

    input_columns_count = x.shape[1]
    output_columns_count = y.shape[1]

    comon_result_properties = {
        "servers_hostnames": selected_servers_names_raw,
        "metadata": metadata,
        "input_shape": x.shape,
        "output_shape": y.shape,
    }

    # Scale values
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    all_non_scaled_values = np.copy(x)
    all_non_scaled_values = np.append(all_non_scaled_values, y, axis=1)
    scaled_values = scaler.fit_transform(all_non_scaled_values)

    scaled_x, scaled_y = scaled_values[:, :input_columns_count], scaled_values[:, -output_columns_count:]

    additional_result_properties = {
        "scaler": scaler
    }

    # Create dataframes with all data
    complete_data_scaled = pandas.concat([pandas.DataFrame(scaled_x, columns=metadata.get("input")),
                                          pandas.DataFrame(scaled_y, columns=metadata.get("output")),
                                          pandas.DataFrame(timestamps, columns=["timestamp"])],
                                         axis=1)
    complete_data_unscaled = pandas.concat([pandas.DataFrame(x, columns=metadata.get("input")),
                                            pandas.DataFrame(y, columns=metadata.get("output")),
                                            pandas.DataFrame(timestamps, columns=["timestamp"])],
                                           axis=1)

    # Export the complete dataframe to csv
    complete_data_scaled.to_csv(f"{ data_folder_path }/complete_data_scaled.csv")
    complete_data_unscaled.to_csv(f"{ data_folder_path }/complete_data_unscaled.csv")

    csv_export_properties = {
        "complete_data_scaled_path": f"{ data_folder_path }/complete_data_scaled.csv",
        "complete_data_unscaled_path": f"{ data_folder_path }/complete_data_unscaled.csv"
    }

    result = {**comon_result_properties, **additional_result_properties, **csv_export_properties}

    new_result = DataResult(**result)

    return new_result


class DataResult(object):

    def __init__(self, servers_hostnames, metadata, input_shape, output_shape, scaler, complete_data_scaled_path, complete_data_unscaled_path):
        self.servers_hostnames = servers_hostnames
        self.metadata = metadata
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.complete_data_scaled_path = complete_data_scaled_path
        self.complete_data_unscaled_path = complete_data_unscaled_path
        self.scaler = scaler

        self.state = "not_loaded"

    def load_data(self):
        self.scaled_df = pandas.read_csv(self.complete_data_scaled_path, parse_dates=["timestamp"])
        self.unscaled_df = pandas.read_csv(self.complete_data_unscaled_path, parse_dates=["timestamp"])
        self.state = "loaded"

    def split_train_and_test_data(self, train_proportion=0.80, shuffle=True):
        scaled_train, scaled_test, unscaled_train, unscaled_test = \
            train_test_split(self.scaled_df,
                             self.unscaled_df,
                             test_size=1 - train_proportion,
                             shuffle=shuffle)

        self.scaled_train_df = scaled_train
        self.scaled_test_df = scaled_test
        self.unscaled_train_df = unscaled_train
        self.unscaled_test_df = unscaled_test
