import unittest
from src.validation.validation import shift_data, shift_data_ltsm, unscale_input, unscale_output, rescale_input, rescale_output, replace_row_variables, extract_substitutions, collect_substitutions_values, shift_data_ltsm_non_scaled
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json


class TestShiftData(unittest.TestCase):

    def setUp(self):

        self.variables = [
            {
                "name": f"var1_n",
                "server_consumption": f"server-1",
                "shift": False,
                "rescale": lambda x: x
            },
            {
                "name": f"var1_n-1",
                "server_temperature": f"server-1",
                "shift": True,
                "shift_count": 1,
                "rescale": lambda x: x
            },
            {
                "name": f"var2_n",
                "server_consumption": f"server-2",
                "shift": False,
                "rescale": lambda x: x
            },
            {
                "name": f"var2_n-1",
                "server_temperature": f"server-2",
                "shift": True,
                "shift_count": 1,
                "rescale": lambda x: x
            },
            {
                "name": f"var2_n-2",
                "server_temperature": f"server-2",
                "shift": True,
                "shift_count": 2,
                "rescale": lambda x: x
            },
            {
                "name": f"output_n",
                "server_temperature": f"server-1",
                "rescale": lambda x: x,
                "output": True
            },
        ]

        pass

    def test_scale_unscale(self):

        non_scaled_array = np.array([
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 0],
        ])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(non_scaled_array)

        non_scaled_array_for_test = np.array([
            [0.5, 1, 3, 4, 3, 3]
        ])

        non_scaled_input_array = non_scaled_array_for_test[:, :5]
        non_scaled_output_array = non_scaled_array_for_test[:, 5:]

        scaled_input = rescale_input(non_scaled_input_array, scaler, self.variables)
        scaled_output = rescale_output(non_scaled_output_array, scaler, self.variables)

        self.assertTrue(np.allclose(scaled_input, np.array([
            [0.5, 0.5, 1, 1, 0.6]
        ])))
        self.assertTrue(np.allclose(scaled_output, np.array([
            [0.5]
        ])))

        unscaled_input = unscale_input(scaled_input, scaler, self.variables)
        unscaled_output = unscale_output(scaled_output, scaler, self.variables)

        self.assertTrue(np.allclose(unscaled_input, np.array([
            [0.5, 1, 3, 4, 3]
        ])))
        self.assertTrue(np.allclose(unscaled_output, np.array([
            [3]
        ])))

    def test_scale_unscale_ltsm(self):

        non_scaled_array = np.array([
            [1, 0, 3, 0, 5, 0],
            [0, 2, 0, 4, 0, 6],
        ])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(non_scaled_array)

        non_scaled_input_array = non_scaled_array[:, :5]
        non_scaled_output_array = non_scaled_array[:, 5:]

        scaled_input = rescale_input(non_scaled_input_array, scaler, self.variables)
        scaled_output = rescale_output(non_scaled_output_array, scaler, self.variables)

        self.assertTrue(np.allclose(scaled_input, np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0]
        ])))
        self.assertTrue(np.allclose(scaled_output, np.array([
            [0],
            [1]
        ])))

        unscaled_input = unscale_input(scaled_input, scaler, self.variables)
        unscaled_output = unscale_output(scaled_output, scaler, self.variables)

        self.assertTrue(np.allclose(unscaled_input, np.array([
            [1, 0, 3, 0, 5],
            [0, 2, 0, 4, 0]
        ])))
        self.assertTrue(np.allclose(unscaled_output, np.array([
            [0],
            [6]
        ])))

    def test_scale_scaling_input_and_outputs(self):

        non_scaled_array = np.array([
            [2, 1, 3],
            [0, 0, 0],
        ])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(non_scaled_array)

        scaled_array = scaler.transform(non_scaled_array)
        unscaled_array = scaler.inverse_transform(scaled_array)

        self.assertTrue(np.array_equal(non_scaled_array, unscaled_array))

    def test_extract_horizontal_substitutions(self):

        expected_response = {
            "var1_n-1": "var1_n",
            "var2_n-2": "var2_n-1",
            "var2_n-1": "var2_n",
        }

        result = extract_substitutions(self.variables)

        dump_expected_response = json.dumps(expected_response, sort_keys=True, indent=2)
        dump_result = json.dumps(result, sort_keys=True, indent=2)

        self.assertTrue(dump_expected_response == dump_result)

    def test_collect_substitutions(self):

        row = np.array([1, 2, 3, 4, 5])

        substitutions = extract_substitutions(self.variables)

        expected_response = {
            "var1_n-1": 1,
            "var2_n-2": 4,
            "var2_n-1": 3,
        }

        result = collect_substitutions_values(row, self.variables, substitutions)

        self.assertTrue(len(result.keys()) == len(expected_response.keys()))

        for key in result.keys():
            self.assertTrue(result[key] == expected_response[key])

    def test_replace_row_variables(self):

        new_values = {
            "var1_n": 100,
            "var1_n-1": 101,
        }

        non_scaled_array = np.array([2, 1, 3])

        result = replace_row_variables(non_scaled_array, self.variables, new_values)

        expected_array = np.array([100, 101, 3])

        self.assertTrue(np.array_equal(expected_array, result))

    def test_shift_data_non_scaled_ltsm(self):

        rows = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])

        initial_substitutions = {
            "var1_n": 38,
            "var2_n": 39
        }

        result = shift_data_ltsm_non_scaled(rows, self.variables, initial_substitutions)

        expected_result = np.array([
            [38.0, 1.0, 39.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ])

        self.assertTrue(np.allclose(result, expected_result))

    def test_shift_data_ltsm(self):

        rows = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])

        initial_substitutions = {
            "var1_n": 38,
            "var2_n": 39
        }

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(np.insert(rows, [5], -10, axis=1))

        scaled_rows = rescale_input(rows, scaler, self.variables)

        scaled_result = shift_data_ltsm(scaled_rows, self.variables, scaler, initial_substitutions)

        unscaled_result = unscale_input(scaled_result, scaler, self.variables)

        expected_result = np.array([
            [38.0, 1.0, 39.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ])

        self.assertTrue(np.allclose(unscaled_result, expected_result))


    def test_shift_data(self):

        rows = np.array([
            [1, 2, 3, 4, 5]
        ])

        initial_substitutions = {
            "var1_n": 38,
            "var2_n": 39
        }

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(np.insert(rows, [5], -10, axis=1))

        scaled_rows = rescale_input(rows, scaler, self.variables)

        scaled_result = shift_data(scaled_rows, self.variables, scaler, initial_substitutions)

        unscaled_result = unscale_input(scaled_result, scaler, self.variables)

        expected_result = np.array([
            [38.0, 1.0, 39, 3.0, 4.0]
        ])

        self.assertTrue(np.allclose(unscaled_result, expected_result))


if __name__ == '__main__':
    unittest.main()
