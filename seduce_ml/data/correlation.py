import pandas as pd
from seduce_ml.data.seduce_data_loader import generate_real_consumption_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas


def investigate_correlations(start_date=None,
                             end_date=None,
                             show_progress=True,
                             data_file_path=None,
                             group_by=60,
                             scaler=None,
                             use_scaler=True,
                             servers_ids=[],
                             figure_folder=None):

    consumption_data = generate_real_consumption_data(
            start_date,
            end_date,
            show_progress,
            data_file_path,
            group_by,
            scaler,
            # False,
            server_ids=servers_ids,
            learning_method="neural"
        )

    x = consumption_data.get("unscaled_x")
    y = consumption_data.get("unscaled_y")

    metadata = consumption_data.get("metadata")
    variable_names = metadata.get("input") + metadata.get("output")

    series = np.append(x, y, axis=1)

    series_as_dataframe = pandas.DataFrame(series, columns=variable_names)

    for output_column in consumption_data.get("metadata").get("output_variables"):
        input_columns = consumption_data.get("metadata").get("input_variables")
        columns = [c.get("name") for c in input_columns] + [output_column.get("name")]
        # Filter columns
        columns = [c for c in columns if "past" not in c]
        series_as_dataframe.to_csv(f"{output_column.get('name')}_export.csv",
                                   columns=columns)

    all_columns = [c.get("name") for c in consumption_data.get("metadata").get("input_variables") + consumption_data.get("metadata").get("output_variables")]
    series_as_dataframe.to_csv(f"all_columns_export.csv",
                               columns=all_columns)

    return True
