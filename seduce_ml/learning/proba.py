from seduce_ml.data.seduce_data_loader import get_additional_variables
from seduce_ml.oracle.oracle import Oracle
from seduce_ml.data.scaling import *
from scipy import interpolate
import numpy

NB_NEIGHBOURS = 10


class ProbaOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def train(self, data):
        self.interpolation_delta = 0.01
        self.interpolation_kind = "cubic"
        self.step = 0.15
        self.data = data

        # Indexes values of all variables
        index = {}
        variables = data.get("metadata").get("variables")
        x_train = data.get("x_train")
        y_train = data.get("y_train")

        input_variables = [var for var in variables if not var.get("output", False)]
        output_variables = [var for var in variables if var.get("output", False)]
        for idx, variable in enumerate(input_variables):
            var_index = {}
            step = self.step

            for idx2, output_variable in enumerate(output_variables):
                var_index2 = {
                    "var_pov": {},
                }
                tuples = [(a, b)
                          for (a, b) in zip(numpy.round(x_train[:, idx], decimals=2),
                                            numpy.round(y_train[:, idx2], decimals=2))]

                # Construct PDF from the <input variable> point of view
                for i in numpy.arange(0, 1.0, step):
                    tuples_in_range = [(a, b) for (a, b) in tuples if a >= i and a < (i + step)]
                    if len(tuples_in_range) == 0:
                        continue
                    key = numpy.round(i, decimals=2)
                    var_index2["var_pov"][key] = {}
                    for j in numpy.arange(0, 1.0, step):
                        key2 = numpy.round(j, decimals=2)
                        tuples_below_value = [(a, b) for (a, b) in tuples_in_range if b <= j]
                        var_index2["var_pov"][key][key2] = 1.0 * len(tuples_below_value) / len(tuples_in_range)
                # Intepolation 2D
                var_pov = var_index2.get("var_pov")
                x = list(var_pov.keys())
                y = list(var_pov[list(var_pov.keys())[0]].keys())
                z = np.array([
                    v1
                    for k,v in var_pov.items()
                    for k1, v1 in v.items()
                ]).reshape(len(y), len(x))
                z_interp = interpolate.interp2d(x, y, z, kind=self.interpolation_kind)
                xnew = np.arange(0, 1, self.interpolation_delta)
                ynew = np.arange(0, 1, self.interpolation_delta)
                znew = z_interp(xnew, ynew)
                var_index2["var_pov_interpolation_2d"] = znew
                # Save data
                var_index[idx2] = var_index2
            index[idx] = var_index
        self.var_index = index

        self.state = "TRAINED"

    def predict(self, unscaled_input_values):
        rescaled_x_input = rescale_input(unscaled_input_values.reshape(1, len(self.metadata.get("input"))), self.scaler, self.metadata.get("variables"))

        variables = self.metadata.get("variables")
        input_variables = [var for var in variables if not var.get("output", False)]
        output_variables = [var for var in variables if var.get("output", False)]

        candidates = []
        for idx, variable in enumerate(input_variables):
            variable_value = rescaled_x_input[0, idx]
            rounded_variable_value = min(np.around(variable_value, decimals=3), 1.0 - self.interpolation_delta)
            x_index = np.arange(0, 1, self.interpolation_delta)
            rounded_variable_value_idx = np.nonzero(x_index >= rounded_variable_value)[0][0]
            for idx2, output_variable in enumerate(output_variables):
                var_pov = self.var_index[idx][idx2].get("var_pov_interpolation_2d")

                pdf = var_pov[rounded_variable_value_idx, :]
                try:
                    equiprobability_idx = np.nonzero(pdf >= 0.45)[0][0]
                except:
                    equiprobability_idx = len(x_index) - 1

                computed_value = x_index[equiprobability_idx]

                candidates += [computed_value]

        reshaped_candidates = np.array(candidates).reshape(len(input_variables), len(output_variables))

        raw_result = numpy.mean(reshaped_candidates, axis=0).reshape(1, len(self.metadata.get("output")))
        rescaled_result = unscale_output(raw_result, self.scaler, self.metadata.get("variables"))
        return rescaled_result.reshape(1, len(self.metadata.get("output")))
