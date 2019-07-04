import numpy as np
import requests
from functional import seq
import os
import json
import calendar
import time
import functools

NORMALIZATION_SERVER = 1000
NORMALIZATION_COOLING = 20000


def normalize_server(x):
    return x / NORMALIZATION_SERVER
    # return x


def normalize_cooling(x):
    # return x / NORMALIZATION_COOLING
    return x / 70.0
    # return x


def normalize_temperature(x):
    return x / 70.0
    # return x


def denormalize_temperature(x):
    return x * 70.0
    # return x


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


def generate_real_consumption_data(start_date=None, end_date=None, show_progress=True):

    if start_date is None:
        start_date = "2019-05-24T08:00:00.000Z"
        # start_date = "2019-06-29T06:00:00.000Z"
    if end_date is None:
        # end_date = "2019-06-11T08:00:00.000Z"
        end_date = "2019-07-04T09:09:35.000Z"

    # Group node data every 120 minutes
    group_by = 2 * 60

    resp = requests.get("https://api.seduce.fr/power_infrastructure/description/tree")
    servers_names_raw = resp.json()['children'][0]['children'][1]['children'][1]['node'].get("children")
    servers_names_raw = sorted(servers_names_raw, key=lambda x: int(x.split("-")[1]))

    servers_names = seq(servers_names_raw).map(lambda x: x.replace("-", "_"))
    # servers_names = servers_names.take(2)

    # nodes_names = servers_names[:1] + ["back_temperature"]
    nodes_names = servers_names + ["back_temperature"]

    data_file_path = "data.json"
    reload_data = False
    # reload_data = True
    if not os.path.exists(data_file_path):
        reload_data = True
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)
            if data.get("start_date") != start_date or data.get("end_date") != end_date:
                reload_data = True

    if reload_data:

        data = {
            "consumptions": {},
            "start_date": start_date,
            "end_date": end_date
        }

        epoch_times = [calendar.timegm(time.strptime(t, '%Y-%m-%dT%H:%M:%S.000Z')) for t in [start_date, end_date]]
        start_epoch = min(epoch_times)
        end_epoch = max(epoch_times)

        temperature_data_url = """https://dashboard.seduce.fr/sensor_data/28b8fb2909000003/aggregated/minutely?start_date=%ss&end_date=%ss""" % (
        start_epoch, end_epoch)
        temperature_data = requests.get(temperature_data_url).json()

        converted_timestamps = [calendar.timegm(time.strptime(x, '%Y-%m-%dT%H:%M:%SZ')) for x in temperature_data.get("timestamps")[:-1]]
        ziped_temperature_data = list(zip(converted_timestamps, temperature_data.get("means")[:-1]))

        group_bytime_ranges_starts = list(range(min(converted_timestamps), max(converted_timestamps), group_by * 60))

        skipped_epoch = []

        # get consumptions of servers
        for node_name in nodes_names:

            if node_name != "back_temperature":
                # node_consumption_url = "https://dashboard.seduce.fr/multitree_sensor_data/%s/aggregated/hourly?start_date=%s&end_date=%s&zoom_ui=true" % (node_name, start_date, end_date)
                node_consumption_url = "https://dashboard.seduce.fr/multitree_sensor_data/%s/aggregated/minutely?start_date=%s&end_date=%s&zoom_ui=true" % (node_name, start_date, end_date)
                current_node_data = requests.get(node_consumption_url).json()

            if node_name not in data.get("consumptions"):

                node_timestamps = current_node_data.get("timestamps")[:-1]
                nodes_converted_timestamps_to_epoch = [calendar.timegm(time.strptime(x, '%Y-%m-%dT%H:%M:%SZ')) for x in node_timestamps]
                node_means = current_node_data.get("means")[:-1]

                node_ziped_data = list(zip(nodes_converted_timestamps_to_epoch, node_means))
                aggregated_node_ziped_data = []

                # for i in range(0, int(len(node_ziped_data) / group_by)):
                idx = 0

                back_temperature_data_structure = None
                back_temperature_data_structure_timestamps = []

                for start_epoch in group_bytime_ranges_starts:

                    if start_epoch in skipped_epoch:
                        continue

                    data_in_current_range = [x for x in node_ziped_data if x[0] >= start_epoch and x[0] <= start_epoch + group_by * 60]

                    # # Get the average temperature in the current interval
                    end_epoch = start_epoch + group_by * 60
                    corresponding_temperature_readings = [t2[1] for t2 in ziped_temperature_data if t2[0] >= start_epoch and t2[0] <= end_epoch]
                    average_temperature = np.mean(corresponding_temperature_readings)

                    if node_name != "back_temperature":
                        mean_current_range = np.mean([tuple2[1] for tuple2 in data_in_current_range])
                    else:

                        if back_temperature_data_structure is None or end_epoch > max(back_temperature_data_structure_timestamps):
                            new_ts = []
                            extra_fetch_size = 1000
                            if back_temperature_data_structure is None:
                                back_temperature_data_structure = average_temperature_aggregated_by_minute(start_epoch, end_epoch + group_by * extra_fetch_size, "back")
                                new_ts = back_temperature_data_structure.get("timestamps")
                            else:
                                update_data_structure = average_temperature_aggregated_by_minute(start_epoch, end_epoch + group_by * extra_fetch_size, "back")
                                for key in ["maxs", "means", "medians", "mins", "stddevs", "timestamps"]:
                                    back_temperature_data_structure[key] += update_data_structure[key]
                                new_ts = update_data_structure.get("timestamps")
                            back_temperature_data_structure_timestamps += [calendar.timegm(time.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')) for ts in new_ts]
                            print("_", end="")
                        else:
                            print("-", end="")

                        data_in_current_range = [x[1] for x in zip(back_temperature_data_structure_timestamps, back_temperature_data_structure.get("maxs")) if x[0] >= start_epoch and x[0] <= start_epoch + group_by * 60]
                        mean_current_range = np.mean(data_in_current_range)
                        if node_name != "back_temperature":
                            print("*", end="")

                    if average_temperature is None or mean_current_range is None or np.isnan(mean_current_range) or np.isnan(average_temperature):
                        skipped_epoch += [start_epoch]
                        continue

                    aggregated_node_ziped_data += [(start_epoch, mean_current_range, idx, average_temperature)]

                    idx += 1

                timestamps = []
                means = []
                idxs = []
                temperatures = []
                for (x, y, z, temp) in aggregated_node_ziped_data:
                    if x is not None and y is not None:
                        timestamps += [x]
                        means += [y]
                        idxs += [z]
                        temperatures += [temp]
                data.get("consumptions")[node_name] = {
                    "timestamps": timestamps,
                    "means": means,
                    "idxs": idxs,
                    "temperatures": temperatures
                }
            if show_progress:
                print('.', end='')

        with open(data_file_path, "w+") as data_file:
            json.dump(data, data_file)
    else:
        with open(data_file_path, "r") as data_file:
            data = json.load(data_file)

    timestamps_with_all_data = None
    for k, v in data.get("consumptions").items():
        if timestamps_with_all_data is None:
            timestamps_with_all_data = v.get("timestamps")
        old_length = len(timestamps_with_all_data)
        timestamps_with_all_data = [item for item in timestamps_with_all_data if item in v.get("timestamps")]
        # print("length %s => %s [%s]" % (old_length, len(timestamps_with_all_data), len(v.get("timestamps"))))

    visualization_data = nodes_names \
        .map(lambda x: data.get("consumptions")[x]) \
        .map(lambda c: zip(c["timestamps"], c["means"], c["temperatures"])) \
        .map(lambda z: seq(z)
             .filter(lambda z: z[0] in timestamps_with_all_data))

    v = visualization_data.map(lambda z: seq(z)
                               .map(lambda z: z[1])
                               .map(lambda x: x if x is not None else 0))

    temperatures = visualization_data.map(lambda z: seq(z)
                                          .map(lambda z: z[2])
                                          .map(lambda x: x if x is not None else 0))[0]

    x = np.transpose(np.array(
        v.take(servers_names.size()).map(lambda x: x.map(lambda z: normalize_server(z)).to_list()).to_list()))
    y = np.array(
        v.drop(servers_names.size()).map(lambda x: x.map(lambda z: normalize_cooling(z)).to_list()).to_list())[0]

    z = np.array(temperatures.map(lambda x: normalize_temperature(x)).to_list()).reshape(len(x), 1)

    timestamps_labels = timestamps_with_all_data
    x_with_temperature = np.hstack((x, z))
    # x_with_temperature = np.hstack((np.sum(x, axis=1).reshape(len(x), 1), z))

    return x_with_temperature, y, timestamps_labels
