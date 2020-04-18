import unittest
import os
from tests.utils import FakeOracle, generate_fake_data
from tests import DATA_TEST_FOLDER, FIGURE_TEST_FOLDER
from seduce_ml.data.data_from_api import generate_real_consumption_data
import shutil
import pandas


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(f"{DATA_TEST_FOLDER}"):
            os.makedirs(f"{DATA_TEST_FOLDER}")
        if not os.path.exists(f"{FIGURE_TEST_FOLDER}"):
            os.makedirs(f"{FIGURE_TEST_FOLDER}")
        self.fake_data = generate_fake_data()
        self.fake_oracle = FakeOracle(self.fake_data.scaler, self.fake_data.metadata, {})

    def tearDown(self) -> None:
        if os.path.exists(f"{DATA_TEST_FOLDER}"):
            shutil.rmtree(f"{DATA_TEST_FOLDER}")
        if os.path.exists(f"{FIGURE_TEST_FOLDER}"):
            shutil.rmtree(f"{FIGURE_TEST_FOLDER}")

    def _find_mock_data_folder(self):
        for dirpath, dirnames, filenames in os.walk("."):
            if "mock_data" in dirnames:
                return os.path.join(dirpath, "mock_data")
        return None

    def test_predict_nsteps_in_future(self):

        start_date = "2020-02-01T00:00:00.000Z"
        end_date = "2020-02-01T02:00:00.000Z"
        data_folder_path = f"{DATA_TEST_FOLDER}"
        group_by = 30
        use_scaler = True
        server_ids = ["ecotype-37"]
        learning_method = "neural"

        last_data = generate_real_consumption_data(start_date,
                                                   end_date,
                                                   data_folder_path=data_folder_path,
                                                   group_by=group_by,
                                                   use_scaler=use_scaler,
                                                   server_ids=server_ids,
                                                   learning_method=learning_method)

        last_data.load_data()

        MOCK_DATA_FOLDER = self._find_mock_data_folder()

        mock_data_scaled_df = pandas.read_csv(f"{ MOCK_DATA_FOLDER }/complete_data_scaled.csv", parse_dates=["timestamp"])
        mock_data_unscaled_df = pandas.read_csv(f"{ MOCK_DATA_FOLDER }/complete_data_unscaled.csv", parse_dates=["timestamp"])

        pandas.testing.assert_frame_equal(mock_data_scaled_df, last_data.scaled_df)
        pandas.testing.assert_frame_equal(mock_data_unscaled_df, last_data.unscaled_df)

        self.assertTrue(last_data is not None)
