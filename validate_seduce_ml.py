from keras.models import load_model
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from itertools import chain

from lib.data.seduce_data_loader import NORMALIZATION_COOLING, NORMALIZATION_SERVER, generate_real_consumption_data, cluster_average_back_temperature

# ORACLE_DUMP_PATH = "'/Users/jonathan/seduceml3.h5'"
# ORACLE_DUMP_PATH = "data/seduceml_2019_06_17_T_11_28_30.h5"
# ORACLE_DUMP_PATH = "data/seduceml_2019_06_20_T_01_20_39.h5"
# ORACLE_DUMP_PATH = "data/seduceml_2019_06_26_T_15_03_29.h5"

# ORACLE_DUMP_PATH = "data/seduceml_2019_07_02_T_13_16_21.h5"
ORACLE_DUMP_PATH = "data/seduceml_2019_07_04_T_10_05_15.h5"

# EXP_DATA = [
#     # {
# #     "loads_name": "seduce_ml",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560173051.8608134,
# #     "start_epoch": 1560165850.4786859,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 4087.936897577996,
# #     "loads_values": [48, 33, 22, 25, 1, 47, 13, 29, 9, 60, 21, 62, 35, 54, 52, 39, 40, 78, 20, 26, 28, 15, 32, 23, 7,
# #                      30, 33, 45, 18, 6, 27, 14, 50, 35, 37, 9, 17, 18, 42, 38, 16, 22, 5, 67, 12, 10, 29, 43],
# #     "target_temperature": 28.0,
# #     "duration": 7200
# # }, {
# #     "loads_name": "compute_low_top_placement",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560182663.3741097,
# #     "start_epoch": 1560175461.9837677,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 4086.1882765462506,
# #     "loads_values": [1, 9, 13, 17, 21, 25, 29, 33, 37, 42, 48, 60, 5, 9, 14, 18, 22, 26, 29, 33, 38, 43, 50, 62, 6, 10,
# #                      15, 18, 22, 27, 30, 35, 39, 45, 52, 67, 7, 12, 16, 20, 23, 28, 32, 35, 40, 47, 54, 78],
# #     "target_temperature": 28.0,
# #     "duration": 7200
# # }, {
# #     "loads_name": "compute_low_top_placement",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560206004.46062,
# #     "start_epoch": 1560198803.0343487,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 4172.736784345922,
# #     "loads_values": [1, 9, 13, 17, 21, 25, 29, 33, 37, 42, 48, 60, 5, 9, 14, 18, 22, 26, 29, 33, 38, 43, 50, 62, 6, 10,
# #                      15, 18, 22, 27, 30, 35, 39, 45, 52, 67, 7, 12, 16, 20, 23, 28, 32, 35, 40, 47, 54, 78],
# #     "target_temperature": 30.0,
# #     "duration": 7200
# # }, {
# #     "loads_name": "seduce_ml",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560215120.4365628,
# #     "start_epoch": 1560207919.059202,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 4100.759659467554,
# #     "loads_values": [48, 33, 22, 25, 1, 47, 13, 29, 9, 60, 21, 62, 35, 54, 52, 39, 40, 78, 20, 26, 28, 15, 32, 23, 7,
# #                      30, 33, 45, 18, 6, 27, 14, 50, 35, 37, 9, 17, 18, 42, 38, 16, 22, 5, 67, 12, 10, 29, 43],
# #     "target_temperature": 30.0,
# #     "duration": 7200
# # }, {
# #     "loads_name": "compute_low_top_placement",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560252871.06813,
# #     "start_epoch": 1560245669.6931486,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 4311.228780103996,
# #     "loads_values": [4, 8, 13, 17, 21, 24, 28, 33, 37, 42, 48, 57, 4, 9, 14, 18, 22, 25, 29, 34, 38, 43, 51, 60, 5, 10,
# #                      15, 18, 22, 26, 31, 35, 40, 45, 53, 68, 7, 12, 15, 20, 23, 27, 31, 35, 40, 47, 54, 71],
# #     "target_temperature": 30.0,
# #     "duration": 7200
# # },
# #     {
# #     "loads_name": "compute_low_top_placement",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560264531.6482775,
# #     "start_epoch": 1560259730.2106931,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 3676.614390394089,
# #     "loads_values": [1, 3, 6, 8, 11, 14, 17, 21, 25, 30, 35, 46, 1, 4, 6, 9, 12, 14, 18, 21, 25, 31, 38, 50, 2, 5, 7, 9,
# #                      12, 15, 19, 23, 27, 32, 39, 55, 3, 5, 8, 11, 13, 16, 19, 23, 28, 34, 41, 61],
# #     "target_temperature": 30.0,
# #     "duration": 4800
# # }, {
# #     "loads_name": "seduce_ml",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560271191.356196,
# #     "start_epoch": 1560266389.9581366,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 3655.2744514573074,
# #     "loads_values": [34, 14, 46, 9, 23, 6, 12, 17, 8, 30, 38, 16, 19, 8, 50, 27, 18, 11, 6, 11, 21, 41, 5, 14, 23, 15,
# #                      25, 31, 28, 35, 1, 55, 9, 3, 25, 19, 32, 2, 13, 5, 3, 39, 12, 61, 7, 21, 4, 1],
# #     "target_temperature": 30.0,
# #     "duration": 4800
# # },
# #     {
# #     "loads_name": "low_down_20",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560545892.4941554,
# #     "start_epoch": 1560537491.1196961,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 3786.203351219798,
# #     "loads_values": [66, 43, 34, 28, 23, 20, 16, 13, 10, 8, 5, 3, 57, 41, 33, 26, 22, 19, 16, 12, 10, 7, 4, 2, 47, 37,
# #                      31, 26, 22, 18, 15, 11, 9, 6, 4, 1, 46, 36, 29, 25, 21, 17, 14, 11, 8, 6, 3, 1],
# #     "target_temperature": 30.0,
# #     "duration": 8400
# # }, {
# #     "loads_name": "low_down_50",
# #     "nb_nodes": 48,
# #     "target_temperature": 30.0,
# #     "end_epoch": 1560556170.0294578,
# #     "start_epoch": 1560547768.6645565,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "loads_values": [89, 76, 69, 64, 58, 54, 49, 45, 41, 35, 30, 23, 84, 74, 67, 63, 58, 52, 49, 44, 39, 34, 28, 20, 81,
# #                      72, 66, 61, 56, 52, 47, 42, 38, 32, 26, 16, 78, 70, 65, 60, 55, 51, 46, 42, 36, 30, 24, 13],
# #     "cooling_consumption": 4356.76395320197,
# #     "duration": 8400
# # }, {
# #     "loads_name": "top_center_20",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560566429.837934,
# #     "start_epoch": 1560558028.3837528,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "cooling_consumption": 3635.253443095629,
# #     "loads_values": [0, 3, 5, 6, 7, 8, 10, 11, 12, 14, 16, 17, 1, 21, 24, 30, 36, 45, 53, 40, 33, 26, 23, 18, 2, 21, 26,
# #                      30, 37, 52, 63, 42, 34, 27, 23, 19, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 20],
# #     "duration": 8400,
# #     "target_temperature": 30.0
# # }, {
# #     "duration": 8400,
# #     "loads_name": "low_top_50",
# #     "nb_nodes": 48,
# #     "end_epoch": 1560576822.459495,
# #     "start_epoch": 1560568421.0832772,
# #     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
# #                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
# #                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
# #                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
# #                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
# #                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
# #                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
# #     "loads_values": [10, 24, 32, 36, 41, 46, 50, 56, 60, 64, 71, 79, 16, 26, 32, 38, 43, 47, 52, 57, 61, 66, 72, 80, 18,
# #                      28, 34, 39, 43, 49, 53, 58, 62, 68, 74, 84, 22, 29, 35, 40, 45, 50, 54, 58, 63, 70, 76, 89],
# #     "cooling_consumption": 4268.645079918941,
# #     "target_temperature": 30.0
# # },
#
#   {
#     "loads_name": "low_top_20",
#     "nb_nodes": 48,
#     "end_epoch": 1560587016.6099567,
#     "start_epoch": 1560578615.2029774,
#     "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15",
#                    "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21",
#                    "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28",
#                    "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34",
#                    "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40",
#                    "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
#                    "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
#     "cooling_consumption": 4008.611062631949,
#     "loads_values": [0, 3, 6, 8, 11, 14, 17, 21, 25, 30, 36, 44, 1, 4, 6, 9, 12, 14, 18, 21, 25, 31, 39, 49, 2, 4, 7,
#                      10, 13, 15, 19, 23, 27, 33, 39, 57, 3, 5, 8, 10, 13, 16, 19, 24, 28, 34, 42, 60],
#     "target_temperature": 30.0,
#     "duration": 8400
# },{
# 	"target_temperature": 30.0,
# 	"start_epoch": 1560794340.0763886,
# 	"cooling_consumption": 4106.263646297599,
# 	"duration": 8400,
# 	"end_epoch": 1560802741.4697516,
# 	"loads_name": "optimized_25_4",
# 	"loads_values": [28, 18, 11, 40, 17, 26, 3, 56, 23, 23, 52, 15, 7, 33, 13, 65, 12, 27, 46, 50, 27, 40, 3, 20, 6, 24, 22, 35, 15, 14, 1, 12, 10, 8, 37, 43, 10, 31, 19, 34, 20, 44, 16, 5, 30, 34, 64, 8],
# 	"nb_nodes": 48,
# 	"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47", "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"]
# }, {
# 	"target_temperature": 30.0,
# 	"start_epoch": 1560804541.9952085,
# 	"cooling_consumption": 3850.208000769054,
# 	"duration": 4800,
# 	"end_epoch": 1560809343.372132,
# 	"loads_name": "centered_25_1",
# 	"loads_values": [1, 6, 7, 9, 11, 12, 14, 16, 17, 19, 21, 22, 2, 27, 30, 35, 42, 52, 62, 47, 40, 33, 29, 23, 4, 28, 31, 37, 44, 54, 75, 50, 42, 35, 30, 24, 5, 7, 9, 10, 12, 13, 15, 17, 18, 20, 21, 26],
# 	"nb_nodes": 48,
# 	"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47", "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"]
# }, {
# 	"target_temperature": 30.0,
# 	"start_epoch": 1560811271.2839208,
# 	"cooling_consumption": 3824.3371503831418,
# 	"duration": 4800,
# 	"end_epoch": 1560816072.7431,
# 	"loads_name": "optimized_25_4",
# 	"loads_values": [29, 18, 11, 39, 17, 25, 2, 57, 23, 24, 51, 15, 7, 31, 13, 68, 13, 27, 47, 50, 27, 40, 4, 20, 6, 24, 22, 37, 16, 14, 2, 12, 10, 8, 37, 43, 9, 30, 19, 34, 20, 44, 16, 4, 30, 34, 64, 8],
# 	"nb_nodes": 48,
# 	"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47", "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"]
# },
#
#     {
# 	"target_temperature": 30.0,
# 	"start_epoch": 1560818307.6665213,
# 	"cooling_consumption": 3843.328474958949,
# 	"duration": 4800,
# 	"end_epoch": 1560823109.0618224,
# 	"loads_name": "centered_25_2",
# 	"loads_values": [1, 5, 8, 9, 11, 13, 14, 15, 17, 19, 21, 22, 2, 27, 31, 36, 42, 53, 60, 46, 39, 33, 29, 23, 4, 28, 33, 38, 45, 58, 67, 48, 41, 35, 29, 24, 5, 7, 8, 10, 11, 13, 15, 17, 19, 20, 22, 25],
# 	"nb_nodes": 48,
# 	"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47", "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"]
# },
#
# #     {
# # 	"target_temperature": 30.0,
# # 	"start_epoch": 1560824960.2864027,
# # 	"nb_nodes": 48,
# # 	"cooling_consumption": 3763.306087335797,
# # 	"end_epoch": 1560829761.7002735,
# # 	"loads_name": "centered_25_1",
# # 	"loads_values": [1, 5, 8, 9, 11, 13, 14, 15, 17, 19, 21, 22, 2, 27, 31, 36, 42, 53, 60, 46, 39, 33, 29, 23, 4, 28, 33, 38, 45, 58, 67, 48, 41, 35, 29, 24, 5, 7, 8, 10, 11, 13, 15, 17, 19, 20, 22, 25],
# # 	"duration": 4800,
# # 	"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47", "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"]
# # },
#
# #     { # This experience was faulty
# # 	"target_temperature": 30.0,
# # 	"start_epoch": 1560831707.193924,
# # 	"cooling_consumption": 3790.220783559113,
# # 	"duration": 4800,
# # 	"end_epoch": 1560836508.5555103,
# # 	"loads_name": "optimized_25_3",
# # 	"loads_values": [31, 19, 9, 39, 17, 25, 2, 53, 23, 22, 58, 13, 5, 35, 14, 67, 13, 27, 46, 48, 29, 41, 5, 20, 7, 24, 22, 33, 15, 15, 1, 11, 11, 8, 38, 42, 8, 29, 21, 36, 19, 45, 17, 4, 28, 33, 60, 10],
# # 	"nb_nodes": 48,
# # 	"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14", "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2", "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25", "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30", "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36", "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41", "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47", "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"]
# # }
#
# ]

EXP_DATA = [
    # {"node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                         "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                         "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                         "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                         "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                         "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                         "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                         "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #          "loads_values": [31, 19, 9, 39, 17, 25, 2, 53, 23, 22, 58, 13, 5, 35, 14, 67, 13, 27, 46, 48, 29, 41, 5,
    #                           20, 7, 24, 22, 33, 15, 15, 1, 11, 11, 8, 38, 42, 8, 29, 21, 36, 19, 45, 17, 4, 28, 33, 60,
    #                           10], "loads_name": "optimized_25_2", "nb_nodes": 48, "target_temperature": 30.0,
    #          "duration": 4800, "start_epoch": 1560984130.5669813, "cooling_consumption": 3755.5891469726253,
    #          "end_epoch": 1560988931.8211656}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [31, 19, 9, 39, 17, 25, 2, 53, 23, 22, 58, 13, 5, 35, 14, 67, 13, 27, 46, 48, 29, 41, 5,
    #                              20, 7, 24, 22, 33, 15, 15, 1, 11, 11, 8, 38, 42, 8, 29, 21, 36, 19, 45, 17, 4, 28, 33,
    #                              60, 10], "loads_name": "optimized_25_3", "nb_nodes": 48, "target_temperature": 30.0,
    #             "duration": 4800, "start_epoch": 1560990876.138823, "cooling_consumption": 3689.1469899425283,
    #             "end_epoch": 1560995677.5406635}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [1, 5, 8, 9, 11, 13, 14, 15, 17, 19, 21, 22, 2, 27, 31, 36, 42, 53, 60, 46, 39, 33, 29,
    #                              23, 4, 28, 33, 38, 45, 58, 67, 48, 41, 35, 29, 24, 5, 7, 8, 10, 11, 13, 15, 17, 19, 20,
    #                              22, 25], "loads_name": "centered_25_1", "nb_nodes": 48, "target_temperature": 30.0,
    #             "duration": 4800, "start_epoch": 1560997609.2554414, "cooling_consumption": 3707.1459646962235,
    #             "end_epoch": 1561002410.65386}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [31, 19, 9, 39, 17, 25, 2, 53, 23, 22, 58, 13, 5, 35, 14, 67, 13, 27, 46, 48, 29, 41, 5,
    #                              20, 7, 24, 22, 33, 15, 15, 1, 11, 11, 8, 38, 42, 8, 29, 21, 36, 19, 45, 17, 4, 28, 33,
    #                              60, 10], "loads_name": "optimized_25_1", "nb_nodes": 48, "target_temperature": 30.0,
    #             "duration": 4800, "start_epoch": 1561004340.083587, "cooling_consumption": 3828.0948746807153,
    #             "end_epoch": 1561009141.5021589}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [42, 33, 52, 30, 36, 62, 55, 67, 57, 94, 43, 60, 44, 80, 90, 73, 65, 83, 63, 75, 25, 29,
    #                              77, 79, 65, 52, 69, 58, 50, 47, 54, 68, 47, 74, 49, 40, 39, 59, 20, 76, 81, 71, 88, 56,
    #                              64, 86, 70, 85], "loads_name": "optimized_25_1", "nb_nodes": 48,
    #             "target_temperature": 30.0, "duration": 7200, "start_epoch": 1561656492.3930507,
    #             "cooling_consumption": 4568.770279108131, "end_epoch": 1561663693.7946324}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [42, 33, 52, 30, 36, 62, 55, 67, 57, 94, 43, 60, 44, 80, 90, 73, 65, 83, 63, 75, 25, 29,
    #                              77, 79, 65, 52, 69, 58, 50, 47, 54, 68, 47, 74, 49, 40, 39, 59, 20, 76, 81, 71, 88, 56,
    #                              64, 86, 70, 85], "loads_name": "optimized_25_2", "nb_nodes": 48,
    #             "target_temperature": 30.0, "duration": 7200, "start_epoch": 1561665678.175151,
    #             "cooling_consumption": 4516.517134760992, "end_epoch": 1561672879.5434067}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [20, 33, 39, 42, 44, 47, 50, 52, 55, 57, 59, 62, 25, 65, 70, 75, 80, 86, 90, 83, 77, 73,
    #                              68, 63, 29, 67, 71, 76, 81, 88, 94, 85, 79, 74, 69, 64, 30, 36, 40, 43, 47, 49, 52, 54,
    #                              56, 58, 60, 65], "loads_name": "centered_25_1", "nb_nodes": 48,
    #             "target_temperature": 30.0, "duration": 7200, "start_epoch": 1561674855.978445,
    #             "cooling_consumption": 4422.150789773763, "end_epoch": 1561682057.3779612}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [20, 33, 39, 42, 44, 47, 50, 52, 55, 57, 59, 62, 25, 65, 70, 75, 80, 86, 90, 83, 77, 73,
    #                              68, 63, 29, 67, 71, 76, 81, 88, 94, 85, 79, 74, 69, 64, 30, 36, 40, 43, 47, 49, 52, 54,
    #                              56, 58, 60, 65], "loads_name": "centered_25_2", "nb_nodes": 48,
    #             "target_temperature": 30.0, "duration": 7200, "start_epoch": 1561683903.2076964,
    #             "cooling_consumption": 4474.403059154696, "end_epoch": 1561691104.5966043}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [42, 33, 52, 30, 36, 62, 55, 67, 57, 94, 43, 60, 44, 80, 90, 73, 65, 83, 63, 75, 25, 29,
    #                              77, 79, 65, 52, 69, 58, 50, 47, 54, 68, 47, 74, 49, 40, 39, 59, 20, 76, 81, 71, 88, 56,
    #                              64, 86, 70, 85], "loads_name": "optimized_25_3", "nb_nodes": 48,
    #             "target_temperature": 30.0, "duration": 7200, "start_epoch": 1561692951.6275175,
    #             "cooling_consumption": 4384.659263820471, "end_epoch": 1561700153.2525575}, {
    #             "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
    #                            "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
    #                            "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
    #                            "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
    #                            "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
    #                            "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
    #                            "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
    #                            "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
    #             "loads_values": [20, 33, 39, 42, 44, 47, 50, 52, 55, 57, 59, 62, 25, 65, 70, 75, 80, 86, 90, 83, 77, 73,
    #                              68, 63, 29, 67, 71, 76, 81, 88, 94, 85, 79, 74, 69, 64, 30, 36, 40, 43, 47, 49, 52, 54,
    #                              56, 58, 60, 65], "loads_name": "centered_25_3", "nb_nodes": 48,
    #             "target_temperature": 30.0, "duration": 7200, "start_epoch": 1561701973.614489,
    #             "cooling_consumption": 4479.40720956486, "end_epoch": 1561709174.9843483},
            {
                "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
                               "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
                               "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
                               "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
                               "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
                               "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
                               "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
                               "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
                "loads_values": [58, 72, 77, 61, 21, 43, 70, 39, 16, 48, 86, 47, 52, 60, 11, 30, 14, 5, 37, 64, 35, 52,
                                 29, 19, 23, 31, 38, 65, 46, 25, 45, 22, 27, 33, 23, 18, 41, 50, 10, 41, 55, 32, 59, 34,
                                 43, 54, 36, 28], "loads_name": "optimized", "nb_nodes": 48, "target_temperature": 30.0,
                "duration": 10800, "start_epoch": 1562181786.5133653, "cooling_consumption": 4331.011964567287,
                "end_epoch": 1562192587.9360576},
            {
                "node_names": ["ecotype-1", "ecotype-10", "ecotype-11", "ecotype-12", "ecotype-13", "ecotype-14",
                               "ecotype-15", "ecotype-16", "ecotype-17", "ecotype-18", "ecotype-19", "ecotype-2",
                               "ecotype-20", "ecotype-21", "ecotype-22", "ecotype-23", "ecotype-24", "ecotype-25",
                               "ecotype-26", "ecotype-27", "ecotype-28", "ecotype-29", "ecotype-3", "ecotype-30",
                               "ecotype-31", "ecotype-32", "ecotype-33", "ecotype-34", "ecotype-35", "ecotype-36",
                               "ecotype-37", "ecotype-38", "ecotype-39", "ecotype-4", "ecotype-40", "ecotype-41",
                               "ecotype-42", "ecotype-43", "ecotype-44", "ecotype-45", "ecotype-46", "ecotype-47",
                               "ecotype-48", "ecotype-5", "ecotype-6", "ecotype-7", "ecotype-8", "ecotype-9"],
                "loads_values": [5, 16, 19, 22, 23, 27, 29, 31, 33, 35, 37, 39, 10, 43, 48, 54, 60, 70, 77, 64, 58, 52,
                                 46, 41, 11, 45, 50, 55, 61, 72, 86, 65, 59, 52, 47, 41, 14, 18, 21, 23, 25, 28, 30, 32,
                                 34, 36, 38, 43], "loads_name": "centered", "nb_nodes": 48, "target_temperature": 30.0,
                "duration": 10800, "start_epoch": 1562194356.8802783, "cooling_consumption": 4270.924704628421,
                "end_epoch": 1562205158.296887}
]


def load_to_power_consumption(load):
    # power_consumption = 99.2314747103 + 2.05275371*load - 0.01289649*load*load
    power_consumption = 1.45331636e+02 + load * 2.96239188e-01 - 7.69497994e+01 * np.exp(-1 * 7.99954988e-02 * load)
    return power_consumption


if __name__ == "__main__":
    oracle = load_model(ORACLE_DUMP_PATH)

    x = []
    y = []
    c = []
    n = []

    for data in EXP_DATA:

        loads_values = data.get("loads_values")
        power_consumption_interpolated_values = [load_to_power_consumption(load_value) for load_value in loads_values]
        normalised_power_consumption_interpolated_values = [load_value / NORMALIZATION_SERVER for load_value in power_consumption_interpolated_values]

        exp_consumption = data.get("cooling_consumption")
        start_epoch = data.get("start_epoch")
        start_date = datetime.datetime\
            .utcfromtimestamp(start_epoch)\
            .replace(tzinfo=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_epoch = data.get("end_epoch")
        end_date = datetime.datetime\
            .utcfromtimestamp(end_epoch)\
            .replace(tzinfo=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        duration = data.get("duration")
        policy = data.get("loads_name")

        real_power_consumption = generate_real_consumption_data(
            start_date="%s" % (start_date),
            end_date="%s" % (end_date),
            show_progress=False
        )

        colors = cluster_average_back_temperature(start_epoch, end_epoch, side="back") # Class by back temperature
        # colors = cluster_average_back_temperature(start_epoch, end_epoch, side="front") # Class by front temperature
        # colors = list(range(0, 12)) * 4 # Class by position
        # colors = [1] * 12 + [2] * 12 + [3] * 12 + [4] * 12 # Class by position

        ext_temp_during_exp = real_power_consumption[0][0][48]

        test = np.array([normalised_power_consumption_interpolated_values])
        test_with_temperature = np.array([normalised_power_consumption_interpolated_values + [ext_temp_during_exp]])
        prediction = oracle.predict(test_with_temperature)[0][0] * NORMALIZATION_COOLING

        # remove last item which contains the external temperature
        consumption_data = real_power_consumption[0][0][:-1]

        x += loads_values
        y += list([c * NORMALIZATION_SERVER for c in consumption_data])
        c += colors
        n += [pos for pos in range(1, 1 + (len(loads_values)))]

        prediction_real_data = oracle.predict(np.array([real_power_consumption[0][0]]))[0][0]
        denormalized_prediction_real_data = prediction_real_data * NORMALIZATION_COOLING

        stupid_prediction = np.mean(consumption_data) * 48 * NORMALIZATION_SERVER / 6067 * 3786

        average_back_temperature = np.mean(colors)
        max_back_temperature = np.max(colors)

        print("[%s, %s, avg: %sC, max: %sC] pred:%s vs pred2:%s vs exp:%s (stupid: %s) using %s policy" % (
            start_date,
            duration,
            average_back_temperature,
            max_back_temperature,
            prediction,
            denormalized_prediction_real_data,
            exp_consumption,
            stupid_prediction,
            policy))

    # def func(x, a, b, c, d, e):
    #     return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x +e

    def func(x, a, b, c, d):
        return a + b * x - c * np.exp(-d*x)
        # return a + b * np.log(c * x)

    # cm = plt.cm.get_cmap('RdYlBu')
    cm = plt.cm.get_cmap('hot_r')

    sorted_data = sorted(zip(x, y, c), key= lambda x: x[0])
    x_sorted = [s[0] for s in sorted_data]
    y_sorted = [s[1] for s in sorted_data]
    c_sorted = [s[2] for s in sorted_data]

    x_sorted_array = np.array(x_sorted)
    y_sorted_array = np.array(y_sorted)
    c_sorted_array = np.array(c_sorted)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_sorted_array, y_sorted_array)

    p0 = [100, 0.01, 100, 0.01]
    popt, pcov = curve_fit(func, x_sorted_array, y_sorted_array, p0)

    fig, ax = plt.subplots()

    # plt.plot(x_sorted_array, y_sorted_array, 'o', label='original data', c=c_sorted_array)
    scatter = ax.scatter(x_sorted_array, y_sorted_array, c=c_sorted_array, cmap=cm, label='consumption', alpha=0.5)

    ydata = func(x_sorted_array, *popt)
    ax.plot(x_sorted_array, ydata, 'r', label='fitted line')
    print(popt)
    print(pcov)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    plt.legend()

    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))

    fig.colorbar(scatter)
    plt.show()
