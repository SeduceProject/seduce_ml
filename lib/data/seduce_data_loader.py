import numpy as np
import requests
from functional import seq
import os
import json
import calendar
import time
from sklearn.preprocessing import MinMaxScaler


def get_neighbours(index):
    return 42


def get_additional_variables(server_id):

    cluster, server_num_str = server_id.split("-")
    server_num = int(server_num_str)

    return [
        {
            "name": f"ecotype_{ server_num }_past_cons_n",
            "server_consumption": f"ecotype-{ server_num }",
            "shift": False,
            "rescale": lambda x: x
        },
        {
            "name": f"ecotype_{ server_num }_past_temp_n",
            "server_temperature": f"ecotype-{ server_num }",
            "shift": False,
            "rescale": lambda x: x,
            "output": True
        },
        # Temperature
        {
            "name": f"ecotype_{ server_num }_past_temp_n-1",
            "server_temperature": f"ecotype-{ server_num }",
            "shift": True,
            "shift_count": 1,
            "rescale": lambda x: x
        },
        {
            "name": f"ecotype_{ server_num }_past_temp_n-2",
            "server_temperature": f"ecotype-{ server_num }",
            "shift": True,
            "shift_count": 2,
            "rescale": lambda x: x
        },
        # Consumption
        {
            "name": f"ecotype_{ server_num }_past_cons_n-1",
            "server_consumption": f"ecotype-{ server_num }",
            "shift": True,
            "shift_count": 1,
        },
        {
            "name": f"ecotype_{ server_num }_past_cons_n-2",
            "server_consumption": f"ecotype-{ server_num }",
            "shift": True,
            "shift_count": 2,
        },
    ]


def get_additional_variables_multiple_servers(server_ids):
    result = []

    for server_id in server_ids:
        result += get_additional_variables(server_id)

    return result


def generate_real_consumption_data(start_date=None,
                                   end_date=None,
                                   show_progress=True,
                                   data_file_path=None,
                                   group_by=60,
                                   scaler=None,
                                   use_scaler=True,
                                   server_id=None,
                                   server_ids=None):

    if data_file_path is None:
        data_file_path = "data/data_60m.json"

    seduce_infrastructure_tree = requests.get("https://api.seduce.fr/infrastructure/description/tree").json()

    servers_names_raw = [f"ecotype-{i}" for i in range(1, 49)]
    servers_names_raw = sorted(servers_names_raw, key=lambda s: int(s.split("-")[1]))

    # selected_servers_names_raw = [f"ecotype-{i}" for i in range(37, 49)]
    if server_id is not None and server_ids is None:
        selected_servers_names_raw = [server_id]
    elif server_id is None and server_ids is not None:
        selected_servers_names_raw = server_ids
    else:
        raise Exception("Could not figure which server to use")

    variables = get_additional_variables_multiple_servers(selected_servers_names_raw)

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

    for var in variables:
        if "sensor" in var:
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
            new_values = [variables_values[shift_count-1] for x in range(0, shift_count)] + variables_values[0:-shift_count]
            data[var.get("name")] = new_values

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

    metadata = {
        "input": [],
        "output": [],
    }

    for var in variables:

        if var.get("exclude_from_training_data", False):
            continue

        temperatures = data[var.get("name")]
        z = np.array(seq(temperatures).map(lambda x: x if x is not None else 0).to_list()).reshape(len(timestamps_labels), 1)
        # Add values of the additional variable to the 'x' array

        if not var.get("output", False):
            if x is None:
                x = z
            else:
                x = np.append(x, z, axis=1)
            metadata["input"] += [var.get("name")]
        else:
            if y is None:
                y = z
            else:
                y = np.append(y, z, axis=1)
            metadata["output"] += [var.get("name")]

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

        return scaled_x, scaled_y, timestamps_labels, data, scaler, shape, selected_servers_names_raw, metadata
    else:
        return x, y, timestamps_labels, data, None, shape, selected_servers_names_raw, metadata
