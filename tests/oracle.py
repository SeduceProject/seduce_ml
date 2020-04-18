import unittest
import os
from tests.utils import FakeOracle, generate_fake_data
from tests import DATA_TEST_FOLDER, FIGURE_TEST_FOLDER


class TestOracle(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(f"{DATA_TEST_FOLDER}"):
            os.makedirs(f"{DATA_TEST_FOLDER}")
        if not os.path.exists(f"{FIGURE_TEST_FOLDER}"):
            os.makedirs(f"{FIGURE_TEST_FOLDER}")
        self.fake_data = generate_fake_data()
        self.fake_data.load_data()
        self.fake_oracle = FakeOracle(self.fake_data.scaler, self.fake_data.metadata, {})

    def test_predict_nsteps_in_future(self):
        data = self.fake_data.unscaled_df[self.fake_data.metadata.get("input")].to_numpy()[0: 0 + 4 + 1]
        result = self.fake_oracle.predict_nsteps_in_future(data, data.copy(), 3)

        # Ensure that previous values have not been reused but rather replaced by values computed at each step.
        # If the previous values have not been replaced, then result[0][0] == 7.0
        self.assertAlmostEqual(result[0, 0], 7.4)


if __name__ == '__main__':
    unittest.main()
