import uuid
import pandas
from autogluon import TabularPrediction as task
from seduce_ml.oracle.oracle import Oracle


class AutoGluonProcessOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def train(self, data, params):
        self.data = data

        self.train_data = task.Dataset(data.unscaled_df)

        autogluon_dir = f'agModels-predictClass/{uuid.uuid4()}'  # specifies folder where to store trained models
        self.predictor = task.fit(train_data=self.train_data, label=self.metadata.get("output")[0], output_directory=autogluon_dir)

        self.state = "TRAINED"

    def predict(self, unscaled_input_values):
        x_input_df = pandas.DataFrame(unscaled_input_values.reshape(1, len(self.metadata.get("input"))), columns=self.metadata.get("input"))
        y_pred = self.predictor.predict(x_input_df)
        return y_pred.reshape(1, len(self.metadata.get("output")))

    def predict_all(self, unscaled_input_values_array):
        row_count = unscaled_input_values_array.shape[0]
        input_data = pandas.DataFrame(unscaled_input_values_array, columns=self.metadata.get("input"))
        result = self.predictor.predict(input_data).reshape(row_count, len(self.metadata.get("output")))
        return result

    def _predict_nsteps_in_future(self, original_data_arrays, data_arrays, nsteps, n=0):

        if n == 0:
            original_data_arrays = self._clean_past_output_values(original_data_arrays)
            data_arrays = self._clean_past_output_values(data_arrays)

        variables_that_travels = [var for var in self.metadata.get("variables") if var.get("become") is not None]
        step_result = self.predict_all(data_arrays[:, n, :])

        if n < nsteps:
            for var in variables_that_travels:
                substituting_output_var_idx = self.metadata.get("output").index(var.get("name"))
                substituting_input_var_idx = self.metadata.get("input").index(var.get("become"))
                data_arrays[:, n + 1, substituting_input_var_idx] = step_result[:, substituting_output_var_idx]

        if n == nsteps:
            return step_result
        else:
            return self._predict_nsteps_in_future(original_data_arrays, data_arrays, nsteps, n + 1)

    def predict_all_nsteps_in_future(self, rows, nsteps):
        rows = self._clean_past_output_values(rows)
        return self._predict_nsteps_in_future(rows, rows.copy(), nsteps=nsteps)
