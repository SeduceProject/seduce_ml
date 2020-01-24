import unittest
from src.data.seduce_data_loader import generate_real_consumption_data
from src.oracle.oracle import create_and_train_oracle
from sklearn.model_selection import train_test_split
from src.validation.validation import validate_seduce_ml
import os
from tests import DATA_TEST_FOLDER, FIGURE_TEST_FOLDER


class TestPrediction60m(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(f"{DATA_TEST_FOLDER}"):
            os.makedirs(f"{DATA_TEST_FOLDER}")
        if not os.path.exists(f"{FIGURE_TEST_FOLDER}"):
            os.makedirs(f"{FIGURE_TEST_FOLDER}")

    def test_prediction_using_neural_network_and_using_only_consumptions(self):
        start_date = "2019-07-20T00:00:00.000Z"
        end_date = "2019-08-20T00:00:00.000Z"

        validation_start_date = "2019-08-20T01:00:00.000Z"
        validation_end_date = "2019-08-27T22:00:00.000Z"

        learning_method = "neural"

        group_by = 60
        use_scaler = True
        percentile = 80

        params = {
            "epoch": 300,
            "nb_layers": 1,
            "neurons_per_layers": 64,
            "activation_function": "relu"
        }

        additional_variables = [
            *[
                {
                    "name": f"ecotype_{i}_cons",
                    "server_consumption": f"ecotype-{i}",
                    "shift": False
                }
                for i in range(1, 49)
            ],
            *[
                {
                    "name": f"ecotype_{i}_past_temp",
                    "server_temperature": f"ecotype-{i}",
                    "shift": True
                }
                for i in range(37, 49)
            ],
        ]

        # Train the neural network
        x, y, tss, data, scaler, shape, servers_names_raw, metadata = \
            generate_real_consumption_data(start_date,
                                           end_date,
                                           group_by=group_by,
                                           use_scaler=use_scaler,
                                           data_file_path=f"{DATA_TEST_FOLDER}/data_test_60m.json",
                                           additional_variables=additional_variables)

        nb_inputs, nb_outputs = shape

        params["nb_inputs"] = nb_inputs
        params["nb_outputs"] = nb_outputs

        train_proportion = 0.80

        best_rmse = None
        best_rmse_perc = None

        for i in range(0, 10):

            x_train, x_test, y_train, y_test, tss_train, tss_test = train_test_split(x,
                                                                                     y,
                                                                                     tss,
                                                                                     test_size=1 - train_proportion,
                                                                                     shuffle=False)

            (oracle, scaler, plot_data, rmse_perc, rmse) = create_and_train_oracle(learning_method=learning_method,
                                                                                   x_train=x_train,
                                                                                   y_train=y_train,
                                                                                   tss_train=tss_train,
                                                                                   x_test=x_test,
                                                                                   y_test=y_test,
                                                                                   tss_test=tss_test,
                                                                                   scaler=scaler,
                                                                                   params=params,
                                                                                   percentile=percentile, )

            server_id = "ecotype-40"

            x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
                generate_real_consumption_data(validation_start_date,
                                               validation_end_date,
                                               data_file_path=f"{DATA_TEST_FOLDER}/data_validation_60m.json",
                                               group_by=group_by,
                                               scaler=scaler,
                                               use_scaler=use_scaler,
                                               additional_variables=additional_variables)

            # Validate data
            validation_rmse, validation_rmse_perc = \
                validate_seduce_ml(x=x_valid,
                                   y=y_valid,
                                   tss=tss_valid,
                                   server_id=server_id,
                                   learning_method=learning_method,
                                   servers_names_raw=servers_names_raw,
                                   use_scaler=use_scaler,
                                   scaler=scaler,
                                   oracle=oracle,
                                   tmp_figures_folder=f"{FIGURE_TEST_FOLDER}",
                                   figure_label="neural_network_using_consumption_only_60m")

            if best_rmse is None or best_rmse > validation_rmse:
                best_rmse = validation_rmse
                best_rmse_perc = validation_rmse_perc

        self.assertLessEqual(best_rmse, 1.2)
        self.assertLessEqual(best_rmse_perc, 4.0)

    def test_prediction_using_knearest_and_using_past_temperature(self):
        start_date = "2019-07-20T00:00:00.000Z"
        end_date = "2019-08-20T00:00:00.000Z"

        validation_start_date = "2019-08-20T01:00:00.000Z"
        validation_end_date = "2019-08-27T22:00:00.000Z"

        learning_method = "knearest"

        group_by = 60
        use_scaler = True
        percentile = 80

        params = {
            "epoch": 300,
            "nb_layers": 1,
            "neurons_per_layers": 32,
            "activation_function": "relu"
        }

        additional_variables = [
            {
                "name": "ecotype_40_past_temp",
                "server_temperature": "ecotype-40",
                "shift": True
            },
        ]

        # Train the neural network
        x, y, tss, data, scaler, shape, servers_names_raw, metadata = \
            generate_real_consumption_data(start_date,
                                           end_date,
                                           group_by=group_by,
                                           use_scaler=use_scaler,
                                           data_file_path=f"{DATA_TEST_FOLDER}/data_test_60m.json",
                                           additional_variables=additional_variables)

        nb_inputs, nb_outputs = shape

        params["nb_inputs"] = nb_inputs
        params["nb_outputs"] = nb_outputs

        train_proportion = 0.80

        x_train, x_test, y_train, y_test, tss_train, tss_test = train_test_split(x,
                                                                                 y,
                                                                                 tss,
                                                                                 test_size=1 - train_proportion,
                                                                                 shuffle=False)

        (oracle, scaler, plot_data, rmse_perc, rmse) = create_and_train_oracle(learning_method=learning_method,
                                                                               x_train=x_train,
                                                                               y_train=y_train,
                                                                               tss_train=tss_train,
                                                                               x_test=x_test,
                                                                               y_test=y_test,
                                                                               tss_test=tss_test,
                                                                               scaler=scaler,
                                                                               params=params,
                                                                               percentile=percentile, )

        server_id = "ecotype-40"

        x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
            generate_real_consumption_data(validation_start_date,
                                           validation_end_date,
                                           data_file_path=f"{DATA_TEST_FOLDER}/data_validation_60m.json",
                                           group_by=group_by,
                                           scaler=scaler,
                                           use_scaler=use_scaler,
                                           additional_variables=additional_variables)

        # Validate data
        validation_rmse, validation_rmse_perc = \
            validate_seduce_ml(x=x_valid,
                               y=y_valid,
                               tss=tss_valid,
                               server_id=server_id,
                               learning_method=learning_method,
                               servers_names_raw=servers_names_raw,
                               use_scaler=use_scaler,
                               scaler=scaler,
                               oracle=oracle,
                               tmp_figures_folder=f"{FIGURE_TEST_FOLDER}",
                               figure_label="knearested_using_previous_temp_60m")

        self.assertLessEqual(validation_rmse, 0.95)
        self.assertLessEqual(validation_rmse_perc, 3.5)

    def test_prediction_using_neural_network_and_using_past_temperature(self):
        start_date = "2019-07-20T00:00:00.000Z"
        end_date = "2019-08-20T00:00:00.000Z"

        validation_start_date = "2019-08-20T01:00:00.000Z"
        validation_end_date = "2019-08-27T22:00:00.000Z"

        learning_method = "neural"

        group_by = 60
        use_scaler = True
        percentile = 80

        params = {
            "epoch": 300,
            "nb_layers": 1,
            "neurons_per_layers": 32,
            "activation_function": "relu"
        }

        additional_variables = [
            {
                "name": "ecotype_40_past_temp",
                "server_temperature": "ecotype-40",
                "shift": True
            },
        ]

        # Train the neural network
        x, y, tss, data, scaler, shape, servers_names_raw, metadata = \
            generate_real_consumption_data(start_date,
                                           end_date,
                                           group_by=group_by,
                                           use_scaler=use_scaler,
                                           data_file_path=f"{DATA_TEST_FOLDER}/data_test_60m.json",
                                           additional_variables=additional_variables)

        nb_inputs, nb_outputs = shape

        params["nb_inputs"] = nb_inputs
        params["nb_outputs"] = nb_outputs

        train_proportion = 0.80

        x_train, x_test, y_train, y_test, tss_train, tss_test = train_test_split(x,
                                                                                 y,
                                                                                 tss,
                                                                                 test_size=1 - train_proportion,
                                                                                 shuffle=False)

        (oracle, scaler, plot_data, rmse_perc, rmse) = create_and_train_oracle(learning_method=learning_method,
                                                                               x_train=x_train,
                                                                               y_train=y_train,
                                                                               tss_train=tss_train,
                                                                               x_test=x_test,
                                                                               y_test=y_test,
                                                                               tss_test=tss_test,
                                                                               scaler=scaler,
                                                                               params=params,
                                                                               percentile=percentile, )

        server_id = "ecotype-40"

        x_valid, y_valid, tss_valid, data_valid, scaler, shape, servers_names_raw, metadata = \
            generate_real_consumption_data(validation_start_date,
                                           validation_end_date,
                                           data_file_path=f"{DATA_TEST_FOLDER}/data_validation_60m.json",
                                           group_by=group_by,
                                           scaler=scaler,
                                           use_scaler=use_scaler,
                                           additional_variables=additional_variables)

        # Validate data
        validation_rmse, validation_rmse_perc = \
            validate_seduce_ml(x=x_valid,
                               y=y_valid,
                               tss=tss_valid,
                               server_id=server_id,
                               learning_method=learning_method,
                               servers_names_raw=servers_names_raw,
                               use_scaler=use_scaler,
                               scaler=scaler,
                               oracle=oracle,
                               tmp_figures_folder=f"{FIGURE_TEST_FOLDER}",
                               figure_label="neural_network_using_previous_temp_60m")

        self.assertLessEqual(validation_rmse, 0.95)
        self.assertLessEqual(validation_rmse_perc, 3.5)


if __name__ == '__main__':
    unittest.main()
