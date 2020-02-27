from keras.models import Sequential
from keras.layers import Dense, Activation
from seduce_ml.oracle.oracle import Oracle
from seduce_ml.data.scaling import *
import copy
from sklearn.preprocessing import MinMaxScaler


class OneOraclePerOutput(Oracle):

    def __init__(self, scaler, metadata, params, CLASS):
        Oracle.__init__(self, scaler, metadata, params)
        self.CLASS = CLASS

        oracles = []
        for output_variable in metadata.get("output"):
            relevant_input = [v.get("name")
                              for v in metadata.get("variables")
                              if ("based_on" not in v or v.get("based_on") == output_variable)
                              and not v.get("output", False)]

            metadata_copy = metadata.copy()
            metadata_copy["input"] = relevant_input
            metadata_copy["input_variables"] = [v
                                                for idx, v in enumerate(metadata.get("variables"))
                                                if v.get("name") in relevant_input]
            metadata_copy["output"] = [output_variable]
            metadata_copy["output_variables"] = [v
                                                 for idx, v in enumerate(metadata.get("variables"))
                                                 if v.get("name") in output_variable]
            metadata_copy["variables"] = metadata_copy["input_variables"] + metadata_copy["output_variables"]

            local_oracle = self.CLASS(None, metadata_copy, params)
            oracles += [local_oracle]

        self.oracles = oracles

    def train(self, data):
        trasnformations = {
            "input": {
                "unscaled_x": "x",
                "unscaled_x_test": "x_test",
                "unscaled_x_train": "x_train"
            },
            "output": {
                "unscaled_y": "y",
                "unscaled_y_test": "y_test",
                "unscaled_y_train": "y_train"
            }
        }
        # scores = []
        for output_variable, local_oracle in zip(self.metadata.get("output"), self.oracles):
            data_copy = data.copy()
            variables = {
                "input": local_oracle.metadata.get("input"),
                "output": [output_variable]
            }
            for var_kind, var_kind_dict in trasnformations.items():
                matching_columns = [idx
                                    for idx, v_name in enumerate(self.metadata.get(var_kind))
                                    if v_name in variables[var_kind]]
                for var in var_kind_dict:
                    data_copy[var] = data_copy[var][:, matching_columns]
            # Compute scaler
            local_scaler = MinMaxScaler(feature_range=(0, 1))
            all_non_scaled_values = np.copy(data_copy.get("unscaled_x"))
            all_non_scaled_values = np.append(all_non_scaled_values, data_copy.get("unscaled_y"), axis=1)
            local_scaler.fit_transform(all_non_scaled_values)
            local_oracle.scaler = local_scaler
            # Compute scaled_variables
            for var_kind, var_kind_dict in trasnformations.items():
                for from_var, to_var in var_kind_dict.items():
                    if var_kind == "input":
                        data_copy[to_var] = rescale_input(data_copy[from_var], local_scaler, local_oracle.metadata.get("variables"))
                    else:
                        data_copy[to_var] = rescale_output(data_copy[from_var], local_scaler, local_oracle.metadata.get("variables"))
            # Train oracle
            score = local_oracle.train(data_copy)
            # scores += [score]
        # The current scaler is not good!
        # self.scaler = OneScalerPerOutput(self)
        # return np.mean(scores)

    def predict(self, unscaled_input_values):
        result = []
        input_array = unscaled_input_values.reshape(1, len(self.metadata.get("input")))
        for output_variable, local_oracle in zip(self.metadata.get("output"), self.oracles):
            matching_columns = [idx
                                for idx, v in enumerate(local_oracle.metadata.get("variables"))
                                if v.get("name") in local_oracle.metadata.get("input")]
            result += [local_oracle.predict(input_array[:, matching_columns])]
        return np.array(result).reshape(1, len(self.metadata.get("output")))


class SimpleOneOraclePerOutput():

    def __init__(self, oracles):
        self.oracles = oracles

    def predict(self, unscaled_input_values):
        result = []
        for local_oracle in self.oracles:
            local_oracle_input_columns = local_oracle.metadata.get("input")
            input_array = unscaled_input_values[local_oracle_input_columns].to_numpy()
            result += [local_oracle.predict(input_array)]
        return np.array(result).reshape(1, len(self.oracles))
