import time
import numpy as np


EPOCH_COUNT = 3000
BATCH_SIZE = 1000
GROUP_BY = 10
PERCENTILE = 80


NETWORK_PATH = "last"

# # <BEFORE HOLIDAYS>
# start_date = "2019-06-01T00:00:00.000Z"
# end_date = "2019-07-05T00:00:00.000Z"
#
# validation_start_date = "2019-07-05T00:00:00.000Z"
# validation_end_date = "2019-07-12T18:00:00.000Z"
# # </BEFORE HOLIDAYS>

# # <AFTER HOLIDAYS>
# start_date = "2019-06-01T00:00:00.000Z"
# end_date = "2019-08-01T00:00:00.000Z"
#
# validation_start_date = "2019-08-01T00:00:00.000Z"
# validation_end_date = "2019-08-19T18:00:00.000Z"
# # </AFTER HOLIDAYS>

# # <DEBUG AFTER HOLIDAYS>
# start_date = "2019-06-17T00:00:00.000Z"
# end_date = "2019-06-20T00:00:00.000Z"
#
# validation_start_date = "2019-08-17T00:00:00.000Z"
# validation_end_date = "2019-08-19T18:00:00.000Z"
# # # </DEBUG AFTER HOLIDAYS>

# # <DEBUG DATE OK>
# start_date = "2019-06-01T00:00:00.000Z"
# end_date = "2019-08-27T11:00:00.000Z"
#
# validation_start_date = "2019-08-27T12:00:00.000Z"
# validation_end_date = "2019-09-03T12:00:00.000Z"
# # </DEBUG DATE OK>

# <DEBUG DATE>
start_date = "2019-08-20T00:00:00.000Z"
end_date = "2019-08-27T11:00:00.000Z"

validation_start_date = "2019-08-27T12:00:00.000Z"
validation_end_date = "2019-08-31T12:00:00.000Z"
# </DEBUG DATE>

tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))


COMPARISON_PLOT_DATA = []

EPOCHS = [
    500,
    # 1000,
    # 2000,
    # 5000,
]
NB_LAYERS = [
    1,
    # 2,
    # 4,
]
NEURONS_PER_LAYER = [
    # 16,
    32,
    # 64
    # 128,
    # 256,
]

ACTIVATION_FUNCTIONS = [
    # "tanh", # Should try this later
    "relu",
    # "sigmoid",
    # "linear", # Should try this later
    # "softmax",
    # "exponential"
]

SHUFFLE = True

SERVER_ID = "ecotype-40"

USE_SCALER = True
# USE_SCALER = False

# LEARNING_METHOD = "neural"
LEARNING_METHOD = "knearest"
# LEARNING_METHOD = "gaussian"

NEURAL_MINIMAL_RMSE = 0.900
MAX_ITERATION_COUNT = 1

TEST_PARAMS = [
    {
        "epoch": epoch,
        "nb_layers": nb_layers,
        "neurons_per_layers": neurons_per_layers,
        "activation_function": activation_function
    }
    for epoch in EPOCHS
    for nb_layers in NB_LAYERS
    for neurons_per_layers in NEURONS_PER_LAYER
    for activation_function in ACTIVATION_FUNCTIONS
]


def compute_average_consumption(x):
    return np.mean(x)


if LEARNING_METHOD == "knearest":
    ADDITIONAL_VARIABLES = [
        {
            "name": "ecotype_40_past_temp_n-1",
            "server_temperature": "ecotype-40",
            "shift": True,
            "rescale": lambda x: x
        },
        {
            "name": "ecotype_40_past_temp_n-2",
            "server_temperature": "ecotype-40",
            "shift": True,
            "shift_count": 2,
            "rescale": lambda x: x
        },
        # Consumption
        {
            "name": "ecotype_40_past_cons_n-1",
            "server_consumption": "ecotype-40",
            "shift": True
        },
        {
            "name": "ecotype_40_past_cons_n-2",
            "server_consumption": "ecotype-40",
            "shift_count": 2,
            "shift": True
        },
    ]
elif LEARNING_METHOD == "gaussian":
    ADDITIONAL_VARIABLES = [
        {
            "name": "ecotype_40_past_temp_n-1",
            "server_temperature": "ecotype-40",
            "shift": True,
            "rescale": lambda x: x
        },
        # {
        #     "name": "ecotype_40_past_temp_n-2",
        #     "server_temperature": "ecotype-40",
        #     "shift": True,
        #     "shift_count": 2,
        #     "rescale": lambda x: x
        # },
        # {
        #     "name": "ecotype_40_past_temp_n-3",
        #     "server_temperature": "ecotype-40",
        #     "shift": True,
        #     "shift_count": 3,
        #     "rescale": lambda x: x
        # },
        # Consumption
        {
            "name": "ecotype_40_past_cons_n-1",
            "server_consumption": "ecotype-40",
            "shift": True
        },
        {
            "name": "ecotype_40_past_cons_n-2",
            "server_consumption": "ecotype-40",
            "shift_count": 2,
            "shift": True
        },
    ]
elif LEARNING_METHOD == "neural":
    ADDITIONAL_VARIABLES = [
        {
            "name": "ecotype_40_past_temp_n-1",
            "server_temperature": "ecotype-40",
            "shift": True,
            "rescale": lambda x: x
        },
        {
            "name": "ecotype_40_past_temp_n-2",
            "server_temperature": "ecotype-40",
            "shift": True,
            "shift_count": 2,
            "rescale": lambda x: x
        },
        # Consumption
        {
            "name": "ecotype_40_past_cons_n-1",
            "server_consumption": "ecotype-40",
            "shift": True
        },
        {
            "name": "ecotype_40_past_cons_n-2",
            "server_consumption": "ecotype-40",
            "shift_count": 2,
            "shift": True
        },
        # {
        #     "name": "ecotype_40_past_cons_n-3",
        #     "server_consumption": "ecotype-40",
        #     "shift_count": 3,
        #     "shift": True
        # },
        # {
        #     "name": "ecotype_40_past_cons_n-4",
        #     "server_consumption": "ecotype-40",
        #     "shift_count": 4,
        #     "shift": True
        # },
        # {
        #     "name": "ecotype_40_past_cons_n-5",
        #     "server_consumption": "ecotype-40",
        #     "shift_count": 5,
        #     "shift": True
        # },
    ]
else:
    raise Exception(f"I could not understand which learning method should be used ({LEARNING_METHOD})")