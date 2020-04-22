import unittest
import os
from tests.utils import FakeOracle, generate_fake_data
from tests import DATA_TEST_FOLDER, FIGURE_TEST_FOLDER
from seduce_ml.validation.validation import evaluate_prediction_power


class TestValidation(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(f"{DATA_TEST_FOLDER}"):
            os.makedirs(f"{DATA_TEST_FOLDER}")
        if not os.path.exists(f"{FIGURE_TEST_FOLDER}"):
            os.makedirs(f"{FIGURE_TEST_FOLDER}")
        self.fake_data = generate_fake_data()
        self.fake_data.load_data()
        self.fake_oracle = FakeOracle(self.fake_data.scaler, self.fake_data.metadata, {
            "seduce_ml": {
                "group_by": 30
            }
        })

    def test_evaluate_prediction_power(self):
        data = self.fake_data
        oracle = self.fake_oracle

        evaluate_prediction_power(data, "ecotype-1", {}, "tests_data", "title=test_data", True, oracle, group_by=30)

        FILES_THAT_SHOULD_EXIST = [
            "tests_data/ecotype-1_title=test_data_auto_correlation_and_cross_correlation.pdf",
            "tests_data/ecotype-1_title=test_data_prediction_power_step=0.pdf",
            "tests_data/ecotype-1_title=test_data_prediction_power_step=1.pdf",
            "tests_data/ecotype-1_title=test_data_prediction_power_step=2.pdf",
            "tests_data/ecotype-1_title=test_data_prediction_power_step=3.pdf",
            "tests_data/ecotype-1_title=test_data_prediction_power_step=4.pdf",
        ]

        for file_path in FILES_THAT_SHOULD_EXIST:
            self.assertTrue(os.path.exists(file_path))


if __name__ == '__main__':
    unittest.main()
