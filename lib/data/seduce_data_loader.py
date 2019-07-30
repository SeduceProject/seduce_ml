import numpy as np
import requests
from functional import seq
import os
import json
import calendar
import time
from sklearn.preprocessing import MinMaxScaler


ADD_EXTERNAL_TEMPERATURE = True
ADD_RETURN_AIR_TEMPERATURE = True


ADDITIONAL_VARIABLES = [
    {
        "name": "room_temperature",
        "sensor": "28b8fb2909000003"
    },
    {
        "name": "return_air_temperature",
        "sensor": "inrow_unit_return_air_temp"
    },
]


def generate_real_consumption_data(start_date=None,
                                   end_date=None,
                                   show_progress=True,
                                   data_file_path="data.json",
                                   group_by=60,
                                   scaler=None):

    seduce_infrastructure_tree = requests.get("https://api.seduce.fr/infrastructure/description/tree").json()

    power_infrastructure_tree = requests.get("https://api.seduce.fr/power_infrastructure/description/tree").json()
    servers_names_raw = power_infrastructure_tree['children'][0]['children'][1]['children'][1]['node'].get("children")
    # servers_names_raw = [f"ecotype-{i}" for i in range(37, 49)]

    servers_names = seq(servers_names_raw)

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
            "max_temperature": None
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
            raise("Some series are incomplete!")

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
        data["filter_timestamps"] = filter_timestamps

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

    for additional_var in ADDITIONAL_VARIABLES:
        additional_var_dict = data.get("sensors_data")[additional_var.get("sensor")]
        data[additional_var.get("name")] = [b
                                            for (a, b) in zip(additional_var_dict["timestamps"],
                                                              additional_var_dict["means"])
                                            if a not in data.get("filter_timestamps")]

    timestamps_with_all_data = data["timestamps"]

    visualization_data = servers_names \
        .map(lambda x: data.get("consumptions")[x]) \
        .map(lambda c: zip(c["timestamps"], c["means"], c["temperatures"])) \
        .map(lambda z: seq(z)
             .filter(lambda z: z[0] in timestamps_with_all_data))

    v = visualization_data.map(lambda z: seq(z)
                               .map(lambda z: z[1])
                               .map(lambda x: x if x is not None else 0))

    x = np.transpose(np.array(
        v.take(servers_names.size()).map(lambda x: x.map(lambda z: z).to_list()).to_list()))

    raw_values_that_will_be_predicted = [tuple_n
                                         for tuple_n in zip(*[data.get("consumptions")[server_name]["temperatures"]
                                                              for server_name in servers_names
                                                              ]
                                                            )
                                         ]
    y = np.array(raw_values_that_will_be_predicted)

    for additional_var in ADDITIONAL_VARIABLES:
        temperatures = data[additional_var.get("name")]
        z = np.array(seq(temperatures).map(lambda x: x if x is not None else 0).to_list()).reshape(len(x), 1)
        # Add external temperature to the the 'x' array
        x = np.append(x, z, axis=1)

    timestamps_labels = timestamps_with_all_data

    # Scale values
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    input_columns_count = x.shape[1]
    output_columns_count = y.shape[1]

    all_non_scaled_values = np.copy(x)
    all_non_scaled_values = np.append(all_non_scaled_values, y, axis=1)
    scaled_values = scaler.fit_transform(all_non_scaled_values)

    scaled_x, scaled_y = scaled_values[:, :input_columns_count], scaled_values[:, -output_columns_count:]

    shape = (input_columns_count, output_columns_count)
    return scaled_x, scaled_y, timestamps_labels, data, scaler, shape
