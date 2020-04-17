import sys
import matplotlib.pyplot as plt
import os
import time
import dill
from seduce_ml.data.seduce_data_loader import generate_real_consumption_data
from seduce_ml.validation.validation import evaluate_prediction_power
from seduce_ml.oracle.oracle import create_and_train_oracle
from seduce_ml.data.scaling import *
import yaml


def train(params,
          tmp_figures_folder,
          server_id,
          one_oracle_per_output):
    comparison_plot_data = []

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

    oracles = []

    # Train the neural network
    consumption_data = \
        generate_real_consumption_data(params.get("seduce_ml").get("start_date"),
                                       params.get("seduce_ml").get("end_date"),
                                       data_file_path=f"data/data_{ params.get('seduce_ml').get('group_by') }m.json",
                                       group_by=params.get('seduce_ml').get('group_by'),
                                       use_scaler=params.get("seduce_ml").get("use_scaler"),
                                       server_id=server_id,
                                       learning_method=learning_method)

    nb_inputs, nb_outputs = consumption_data.input_shape[1], consumption_data.input_shape[1] + consumption_data.output_shape[1]

    consumption_data.load_data()
    consumption_data.split_train_and_test_data()

    (oracle_object, plot_data, score) =\
        create_and_train_oracle(consumption_data,
                                learning_method=learning_method,
                                metadata=consumption_data.metadata,
                                params=params,
                                one_oracle_per_output=one_oracle_per_output)

    result = evaluate_prediction_power(
        consumption_data,
        server_id=server_id,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=f"train_predict_power_idx",
        produce_figure=False,
        oracle_object=oracle_object
    )

    oracles += [{
        "plot_data": plot_data,
        "score": score,
        # "idx": i,
        "oracle_object": oracle_object
    }]

    sorted_oracles = sorted(oracles, key=lambda oracle: oracle.get("oracle_train_rmse"))
    selected_oracle = sorted_oracles[0]

    best_oracle_object = selected_oracle.get("oracle_object")
    best_plot_data = selected_oracle.get("plot_data")

    param_label = f"{learning_method}"

    epoch = time.time()

    comparison_plot_data += [{
        "epoch": epoch,
        "nb_layers": params.get("nb_layers"),
        "neurons_per_layers": params.get("neurons_per_layers"),
        "activation_function": params.get("activation_function"),
        "tmp_figures_folder": tmp_figures_folder,
        "method": learning_method,
        "score": score
    }]

    # Draw the comparison between actual data and prediction:
    server_idx = consumption_data.servers_hostnames.index(server_id)

    start_step = int(0 * len(best_plot_data))
    end_step = int(1.0 * len(best_plot_data))

    sorted_plot_data = sorted(best_plot_data, key=lambda d: d["x"])[start_step:end_step]

    fig = plt.figure()
    ax = plt.axes()

    x_data = [d["x"] for d in sorted_plot_data]
    y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
    y2_data = [d["temp_pred"][server_idx] for d in sorted_plot_data]
    x_data = range(0, len(y1_data))

    ax.plot(x_data, y1_data, color='blue', label='actual temp.', linewidth=0.5)
    ax.plot(x_data, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)

    plt.legend()

    plt.xlabel('Time (hour)')
    plt.ylabel('Back temperature of %s (deg. C)' % server_id)
    plt.title("%s" % param_label)

    plt.savefig(f"{tmp_figures_folder}/{server_id}_training_{param_label}.pdf"
                .replace(":", " ")
                .replace("'", "")
                .replace("{", "")
                .replace("}", "")
                .replace(" ", "_")
                )

    # x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
    consumption_data_validation =\
        generate_real_consumption_data(params.get('seduce_ml').get('validation_start_date'),
                                       params.get('seduce_ml').get('validation_end_date'),
                                       data_file_path=f"data/data_validation_{ params.get('seduce_ml').get('group_by') }m.json",
                                       group_by=params.get('seduce_ml').get('group_by'),
                                       scaler=consumption_data.scaler,
                                       use_scaler=params.get('seduce_ml').get('use_scaler'),
                                       server_id=server_id,
                                       learning_method=learning_method)

    if learning_method == "neural":
        figure_label = f"validation_{learning_method}__" \
                       f"epoch_{ params.get('configuration').get('neural').get('epoch_count') }__" \
                       f"layers_{ params.get('configuration').get('neural').get('layers_count') }__" \
                       f"neurons_per_layer_{ params.get('configuration').get('neural').get('neurons_per_layers') }__" \
                       f"activ_{ params.get('configuration').get('neural').get('activation_function') }"
    elif learning_method == "knearest":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "gaussian":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "ltsm":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "gradient_boost_regressor":
        figure_label = f"validation_{learning_method}"
    elif learning_method == "autogluon":
        figure_label = f"validation_{learning_method}"
    else:
        raise Exception("Could not understand which learning method is used")

    consumption_data_validation.load_data()

    # Evaluate prediction power
    evaluate_prediction_power(
        consumption_data_validation,
        oracle_object=oracle_object,
        server_id=server_id,
        tmp_figures_folder=tmp_figures_folder,
        figure_label=figure_label
    )

    for oracle_data in sorted_oracles:
        this_score = oracle_data.get("score")
        oracle_idx = oracle_data.get("idx")

        # Evaluate prediction power
        evaluate_prediction_power(
            consumption_data_validation,
            oracle_object=oracle_object,
            server_id=server_id,
            tmp_figures_folder=tmp_figures_folder,
            figure_label=f"evaluate_score_{this_score:.5f}_idx_{oracle_idx}",
            produce_figure=False
        )

    return best_oracle_object


if __name__ == "__main__":

    import multiprocessing as mp;
    mp.set_start_method("spawn")

    print("Hello from seduce ML")

    tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))

    def compute_average_consumption(x):
        return np.mean(x)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

    with open("seduce_ml.yaml") as file:
        PARAMS = yaml.load(file)

        learning_method = PARAMS.get("seduce_ml").get("learning_method")

        if learning_method in ["neural", "ltsm"]:
            params = {
                "epoch_count": PARAMS.get("configuration").get("neural").get("epoch_count"),
                "batch_size": PARAMS.get("configuration").get("neural").get("batch_size"),
                "nb_layers": PARAMS.get("configuration").get("neural").get("layers_count"),
                "neurons_per_layers": PARAMS.get("configuration").get("neural").get("neurons_per_layer"),
                "activation_function": PARAMS.get("configuration").get("neural").get("activation_function"),
            }
        else:
            params = {}

        params["group_by"] = PARAMS.get("seduce_ml").get("group_by")

        server_ids = [f"ecotype-{i}"
                      for i in range(37, 49)]
        # server_ids = [f"ecotype-{i}"
        #               for i in range(37, 38)]

        oracle_objects = []
        for server_id in server_ids:
            oracle_object = train(
                PARAMS,
                tmp_figures_folder,
                server_id,
                False,
            )
            oracle_objects += [oracle_object]

        # Create an aggregate of all oracle objects
        from seduce_ml.learning.one_oracle_per_output import SimpleOneOraclePerOutput
        aggregateOracle = SimpleOneOraclePerOutput(oracle_objects)

        with open('oracle.pickle', 'wb') as oracle_object_file:
            dill.dump(aggregateOracle, oracle_object_file)

        # Check that the model is working
        last_data = generate_real_consumption_data(PARAMS.get("seduce_ml").get("start_date"),
                                                   PARAMS.get("seduce_ml").get("end_date"),
                                                   data_file_path=f"data/data_{PARAMS.get('seduce_ml').get('group_by')}m.json",
                                                   group_by=PARAMS.get("seduce_ml").get("group_by"),
                                                   use_scaler=PARAMS.get("seduce_ml").get("use_scaler"),
                                                   server_ids=server_ids,
                                                   learning_method=learning_method)

        last_data.load_data()
        last_row = last_data.unscaled_df[last_data.metadata.get("input")].iloc[[-1]]

        with open('oracle.pickle', 'rb') as oracle_object_file:
            aggregateOracle = dill.load(oracle_object_file)
            prediction = aggregateOracle.predict(last_row)
            print(prediction)

    sys.exit(0)
