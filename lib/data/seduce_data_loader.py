import numpy as np
import requests
from functional import seq
import os
import json
import calendar
import time
import functools
import math

NORMALIZATION_SERVER = 200
# NORMALIZATION_COOLING = 20000


def simulate_consumption_function(data):

    weight = [
        0.82, 0.85, 0.87, 0.90, 0.97, 0.99, 0.93, 0.82, 0.72, 0.70, 0.85, 0.98,
        0.82, 0.85, 0.87, 0.90, 0.97, 0.99, 0.93, 0.82, 0.72, 0.70, 0.78, 0.85,
        0.95, 0.94, 0.87, 0.90, 0.97, 0.99, 0.93, 0.82, 0.72, 0.70, 0.70, 0.68,
        0.97, 0.99, 0.87, 0.90, 0.97, 0.99, 0.93, 0.82, 0.72, 0.70, 0.70, 0.68,
    ]

    if len(data) < 10:
        raise Exception("Error: len(data) < 0")

    consumption = sum([x * y for (x, y) in zip(weight, data)])
    normalized_consumption = consumption / len(weight)

    # return math.sin(normalized_consumption)
    return normalized_consumption


def generate_fake_consumption_data(nb_servers, nb_data):
    x = np.random.random((nb_data, nb_servers))
    y = np.apply_along_axis(simulate_consumption_function, 1, x)
    return x, y


def cluster_average_temperature(start_epoch, end_epoch, side="back"):

    resp = requests.get("https://api.seduce.fr/infrastructure/description/tree")
    resp_json = resp.json()
    servers_names = list(resp_json.get("power").keys())

    back_temperature_sensors = [v.get(side) for v in resp_json.get("temperature").values() if v.get(side) is not None]

    server_sensors_list = []

    for back_temperature_sensor in back_temperature_sensors:
        sensors = [(s.get("tags")[0], s.get("serie")) for s in back_temperature_sensor.values() if s.get("tags")]

        server_sensors_list += sensors

    sorted_server_sensors_list = sorted(server_sensors_list, key=lambda x: int(x[0].split("-")[-1]))
    sorted_sensors = [x[1] for x in sorted_server_sensors_list]

    temperatures = []

    for sensor in sorted_sensors:
        temperature_url = "https://api.seduce.fr/sensors/%s/measurements?start_date=%s&end_date=%s" % (sensor, int(start_epoch), int(end_epoch))
        sensor_temperatures = requests.get(temperature_url).json().get("values")
        temperatures += [np.mean(sensor_temperatures)]
        print(".", end="")
    print("")

    return temperatures


def average_temperature_aggregated_by_minute(start_epoch, end_epoch, side="back"):
    temperature_url = "https://dashboard.seduce.fr/rack/%s/temperatures/aggregated?start_date=%ss&end_date=%ss" % (side, int(start_epoch), int(end_epoch))
    sensor_temperatures = requests.get(temperature_url).json()

    return sensor_temperatures


def generate_real_consumption_data(start_date=None, end_date=None, show_progress=True, data_file_path="data.json", training_data=None):

    if start_date is None:
        # start_date = "2019-05-24T08:00:00.000Z"
        start_date = "2019-06-01T06:00:00.000Z"
        # start_date = "2019-07-07T06:00:00.000Z"
    if end_date is None:
        # end_date = "2019-06-11T08:00:00.000Z"
        end_date = "2019-07-08T09:09:35.000Z"

    # Group node data every 120 minutes
    group_by = 60
    # group_by = 35
    # group_by = 2 * 60

    seduce_infrastructure_tree = requests.get("https://api.seduce.fr/infrastructure/description/tree").json()

    power_infrastructure_tree = requests.get("https://api.seduce.fr/power_infrastructure/description/tree").json()
    servers_names_raw = power_infrastructure_tree['children'][0]['children'][1]['children'][1]['node'].get("children")
    servers_names_raw = sorted(servers_names_raw, key=lambda x: int(x.split("-")[1]))

    servers_names = seq(servers_names_raw)
    # servers_names = servers_names.take(2)

    # nodes_names = servers_names[:1] + ["back_temperature"]
    # nodes_names = servers_names

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
                "temperatures": dump_data.get("sensors_data")[back_temperature_sensor]["means"],
                "timestamps": dump_data.get("sensors_data")[back_temperature_sensor]["timestamps"],
            }

            if show_progress:
                print('.', end='')

        data["room_temperature"] = data.get("sensors_data")["28b8fb2909000003"]["means"]

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

        # Find normalizaion parameters
        for server_name in servers_names:
            data_server = data["consumptions"][server_name]

            server_min_consumption = min(data_server["means"])
            server_max_consumption = max(data_server["means"])
            server_min_temperature = min(data_server["temperatures"])
            server_max_temperature = max(data_server["temperatures"])

            if data["min_consumption"] is None or data["min_consumption"] > server_min_consumption:
                data["min_consumption"] = server_min_consumption

            if data["max_consumption"] is None or data["max_consumption"] < server_max_consumption:
                data["max_consumption"] = server_max_consumption

            if data["min_temperature"] is None or data["min_temperature"] > server_min_temperature:
                data["min_temperature"] = server_min_temperature

            if data["max_temperature"] is None or data["max_temperature"] < server_max_temperature:
                data["max_temperature"] = server_max_temperature

        min_cons = data["min_consumption"]
        max_cons = data["max_consumption"]
        min_temp = data["min_temperature"]
        max_temp = data["max_temperature"]

        if training_data is not None:
            min_cons = training_data["min_consumption"]
            max_cons = training_data["max_consumption"]
            min_temp = training_data["min_temperature"]
            max_temp = training_data["max_temperature"]

        # Normalize data
        for server_name in servers_names:
            data_server = data["consumptions"][server_name]

            data_server["means"] = [(1.0 * x - min_cons) / (max_cons - min_cons)
                                    for x in data_server["means"]]
            data_server["temperatures"] = [(1.0 * x - min_temp) / (max_temp - min_temp)
                                           for x in data_server["temperatures"]]

        data["room_temperature"] = [tuple_2[1]
                                    for tuple_2 in zip(data["timestamps"], data["room_temperature"])
                                    if tuple_2[0] not in filter_timestamps]

        with open(data_file_path, "w+") as data_file:
            json.dump(data, data_file)
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)

    timestamps_with_all_data = data["timestamps"]

    visualization_data = servers_names \
        .map(lambda x: data.get("consumptions")[x]) \
        .map(lambda c: zip(c["timestamps"], c["means"], c["temperatures"])) \
        .map(lambda z: seq(z)
             .filter(lambda z: z[0] in timestamps_with_all_data))

    v = visualization_data.map(lambda z: seq(z)
                               .map(lambda z: z[1])
                               .map(lambda x: x if x is not None else 0))

    temperatures = data["room_temperature"]

    x = np.transpose(np.array(
        v.take(servers_names.size()).map(lambda x: x.map(lambda z: z).to_list()).to_list()))

    # Compute values that will be predicted
    def select_tuple_n(tuple_n):
        return list(tuple_n)
        # return [tuple_n[23]]
        # return tuple_n[0]
        # return max(tuple_n)
        # return 30000
        # return sum(tuple_n)

    raw_values_that_will_be_predicted = [select_tuple_n(tuple_n)
                                         for tuple_n in zip(*[server_data["temperatures"]
                                                              for server_name, server_data in data.get("consumptions").items()]
                                                            )
                                         ]

    y = np.array(raw_values_that_will_be_predicted)

    timestamps_labels = timestamps_with_all_data

    # # Add external temperature to the the 'x' array
    # min_temp = data["min_temperature"]
    # max_temp = data["max_temperature"]
    # z = np.array(seq(temperatures).map(lambda x: (1.0 * x - min_temp) / (max_temp - min_temp) if x is not None else 0).to_list()).reshape(len(x), 1)
    # x = np.append(x, z, axis=1)

    return x, y, timestamps_labels, data
