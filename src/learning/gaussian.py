from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from src.validation.validation import rescale_input, unscale_input, rescale_output, unscale_output
from src.data.seduce_data_loader import get_additional_variables
import numpy

NB_NEIGHBOURS = 10


class GaussianProcessOracle(object):

    def __init__(self, x_train, y_train, tss_train, scaler, metadata):
        self.x_train = x_train
        self.y_train = y_train
        self.tss_train = tss_train
        self.scaler = scaler
        self.metadata = metadata

        self.variables = get_additional_variables("ecotype-43", "gaussian")

        # # Instantiate a Gaussian Process model
        # param = [5 for x in range(0, x_train.shape[1])]
        #
        # kernel = RBF(param, (1e-2, 1e2))
        # self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-2)
        # self.gp.fit(self.x_train, self.y_train)

        # Index values of all variables
        index = {}
        input_variables = [var for var in self.variables if not var.get("output", False)]
        self.step = 0.11
        for idx, variable in enumerate(input_variables):
            var_index = {
                "var_pov": {},
                "output_pov": {}
            }
            step = self.step
            tuples = [(a, b)
                      for (a, b) in zip(numpy.round(self.x_train[:, idx], decimals=2),
                                        numpy.round(self.y_train[:, 0], decimals=2))]
            # Construct PDF from the <input variable> point of view
            for i in numpy.arange(0, 1.0, step):
                tuples_in_range = [(a, b) for (a, b) in tuples if a >= i and a < (i + step)]
                if len(tuples_in_range) == 0:
                    continue
                key = numpy.round(i, decimals=2)
                var_index["var_pov"][key] = {}
                for j in numpy.arange(0, 1.0, step):
                    key2 = numpy.round(j, decimals=2)
                    tuples_below_value = [(a, b) for (a, b) in tuples_in_range if b <= j]
                    var_index["var_pov"][key][key2] = 1.0 * len(tuples_below_value) / len(tuples_in_range)
            # Construct PDF from the <output variable> point of view
            for i in numpy.arange(0, 1.0, step):
                tuples_in_range = [(a, b) for (a, b) in tuples if b >= i and b < (i + step)]
                if len(tuples_in_range) == 0:
                    continue
                key = numpy.round(i, decimals=2)
                var_index["output_pov"][key] = {}
                for j in numpy.arange(0, 1.0, step):
                    key2 = numpy.round(j, decimals=2)
                    tuples_below_value = [(a, b) for (a, b) in tuples_in_range if a <= j]
                    var_index["output_pov"][key][key2] = 1.0 * len(tuples_below_value) / len(tuples_in_range)
            index[idx] = var_index
        self.var_index = index

    def average_cons_bottom(self, unscaled_x_input):
        return numpy.max(unscaled_x_input[0:3])

    def distance(self, x_input_a, x_input_b):
        avg_cons_a = self.average_cons_bottom(x_input_a)
        avg_cons_b = self.average_cons_bottom(x_input_b)
        return abs(x_input_a[0] - x_input_b[0]) + 0.75 * abs(x_input_a[1] - x_input_b[1]) + 0.1 * abs(x_input_a[1] - x_input_b[1]) + 0.0 * abs(avg_cons_a - avg_cons_b) + 0.9 * abs(x_input_a[6] - x_input_b[6])

    def predict(self, x_input, idx=None):
        # # Make the prediction on the meshed x-axis (ask for MSE as well)
        # # y_pred, sigma = self.gp.predict(x_input, return_std=True)
        #
        # # unscaled_x_input = unscale_input(x_input, self.scaler, self.variables)
        #
        # # average_cons_bottom = self.average_cons_bottom(x_input)
        #
        # points_with_distance = [(self.distance(x_input[0], x), x, y, z)
        #                         for (x, y, z) in zip(self.x_train, self.y_train, self.tss_train)]
        # # close_points = [t4 for t4 in points_with_distance if (np.sqrt((t4[1][0] - x_input[0][0]) ** 2)) < 2]
        # sorted_points = sorted(points_with_distance, key=lambda t4: t4[0])
        # selected_points = [t4[-2] for t4 in sorted_points[0:NB_NEIGHBOURS]]
        # result = numpy.median(selected_points, axis=0)

        expected_value = self.y_train[idx]

        # debug_vector = [(unscale_output(c.reshape(1, 1), self.scaler, self.variables), unscale_input(b.reshape(1, len(self.variables)-1), self.scaler, self.variables)) for (a,b,c,d) in sorted_points[0:100]]
        possible_values = []
        input_variables = [var for var in self.variables if not var.get("output", False)]

        candidates = []
        for idx, variable in enumerate(input_variables):
            # key = numpy.round(x_input[0, idx])
            # possible_values += self.var_index[idx].get(key, {"values": {}}).get("values")

            var_pov = self.var_index[idx].get("var_pov")

            key = min(var_pov.keys())
            for i in var_pov:
                if x_input[0, idx] < i:
                    break
                key = i

            pdf = var_pov.get(key)

            min_value = 0.0
            threshold = 0.5
            for i in pdf:
                if pdf[i] >= threshold:
                    break
                min_value = i

            if pdf[min_value] == 1.0:
                computed_value = min_value
            if pdf[min_value] == 0.5:
                computed_value = min_value
            else:
                if min_value != 0.9:
                    next_key = numpy.round(min_value + self.step, decimals=2)
                    coeff = (pdf[next_key] - pdf[min_value]) / self.step
                else:
                    coeff = (1.0 - pdf[min_value]) / self.step

                computed_value = min_value + (threshold - pdf[min_value]) / coeff
            candidates += [computed_value]

        return [numpy.mean(candidates), 0]
