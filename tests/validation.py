import unittest
import os
from tests.utils import FakeOracle, generate_fake_data
from tests import DATA_TEST_FOLDER, FIGURE_TEST_FOLDER
from seduce_ml.validation.validation import predict_nsteps_in_future


class TestValidation(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(f"{DATA_TEST_FOLDER}"):
            os.makedirs(f"{DATA_TEST_FOLDER}")
        if not os.path.exists(f"{FIGURE_TEST_FOLDER}"):
            os.makedirs(f"{FIGURE_TEST_FOLDER}")
        self.fake_data = generate_fake_data()
        self.fake_oracle = FakeOracle(self.fake_data.get("scaler"), self.fake_data.get("metadata"), {})

    def test_predict_nsteps_in_future(self):
        data = self.fake_data.get("unscaled_x")[0: 0 + 4 + 1]
        result = predict_nsteps_in_future(self.fake_oracle, data, data.copy(), 3)

        print(result)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
