import time

EPOCH_COUNT = 3000
BATCH_SIZE = 1000
GROUP_BY = 60
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

# <DEBUG DATE>
start_date = "2019-07-20T00:00:00.000Z"
end_date = "2019-08-20T00:00:00.000Z"

validation_start_date = "2019-08-20T01:00:00.000Z"
validation_end_date = "2019-08-27T22:00:00.000Z"
# </DEBUG DATE>

tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))


COMPARISON_PLOT_DATA = []

EPOCHS = [
    300,
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
    400,
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

NB_RUN = 5

SERVER_ID = "ecotype-40"

USE_SCALER = True
# USE_SCALER = False

LEARNING_METHOD = "neural"
# LEARNING_METHOD = "knearest"

NEURAL_MINIMAL_RMSE = 0.900
MAX_ITERATION_COUNT = 5

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

ADDITIONAL_VARIABLES = [
    {
        "name": "ecotype_40_past_temp",
        "server_temperature": "ecotype-40",
        "shift": True
    },
    # # # # Consumption
    # {
    #     "name": "ecotype_40_past_cons",
    #     "server_consumption": "ecotype-40",
    #     "shift": True
    # },
    # {
    #     "name": "avg_consumption",
    #     "consumption_function": compute_average_consumption,
    #     "shift": False
    # },
    # {
    #     "name": "past_avg_consumption",
    #     "consumption_function": compute_average_consumption,
    #     "shift": True
    # },
    # *[
    #     {
    #         "name": f"ecotype_{i}_past_cons",
    #         "server_consumption": f"ecotype-{i}",
    #         "shift": True
    #     }
    #     for i in range(1, 49)
    # ],
    # *[
    #     {
    #         "name": f"ecotype_{i}_cons",
    #         "server_consumption": f"ecotype-{i}",
    #         "shift": False
    #     }
    #     for i in range(1, 49)
    # ],
    # *[
    #     {
    #         "name": f"ecotype_{i}_past_temp",
    #         "server_temperature": f"ecotype-{i}",
    #         "shift": True
    #     }
    #     for i in range(37, 49)
    # ],
]