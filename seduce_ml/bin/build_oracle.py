import sys
import os
import time
import dill
from seduce_ml.data.data_from_api import generate_real_consumption_data
import yaml
from seduce_ml.oracle.oracle import build_new_oracle

if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn")

    print("Hello from seduce ML")

    tmp_figures_folder = "tmp/%s" % time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(time.time()))

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(tmp_figures_folder):
        os.makedirs(tmp_figures_folder)

    with open("seduce_ml.yaml") as file:
        PARAMS = yaml.safe_load(file)

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
            oracle_object = build_new_oracle(
                PARAMS,
                tmp_figures_folder,
                server_id
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
                                                   data_folder_path=f"data",
                                                   data_file_name="data_validation_final.json",
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
