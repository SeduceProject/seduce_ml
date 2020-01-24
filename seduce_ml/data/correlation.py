import pandas as pd
from seduce_ml.data.seduce_data_loader import generate_real_consumption_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def investigate_correlations(start_date=None,
                             end_date=None,
                             show_progress=True,
                             data_file_path=None,
                             group_by=60,
                             scaler=None,
                             use_scaler=True,
                             servers_ids=[],
                             figure_folder=None):

    x, y, timestamps_labels, data, scaler, shape, selected_servers_names_raw, metadata =\
        generate_real_consumption_data(
            start_date,
            end_date,
            show_progress,
            data_file_path,
            group_by,
            None,
            False,
            server_ids=servers_ids)

    variable_names = metadata.get("input") + metadata.get("output")

    scaler = MinMaxScaler(feature_range=(0, 1))

    series = np.append(x, y, axis=1)
    # series = series - series.min(axis=0)

    selected_data = series
    scaled_selected_data = scaler.fit_transform(selected_data)

    # Compute difference between rows of the 'selected_data' matrix
    diff_selected_data = selected_data[1:, :] - selected_data[:-1, :]

    index_investigated_temperature = variable_names.index("ecotype_43_temperature")
    filtered_diff_data_t2 = [(idx, row)
                             for idx, row in zip(range(0, len(diff_selected_data.tolist())), diff_selected_data.tolist())
                             if abs(row[index_investigated_temperature]) > 4.00]
    filtered_diff_data = [t2[1] for t2 in filtered_diff_data_t2]
    indexes = [t2[0] for t2 in filtered_diff_data_t2]
    scaled_diff_selected_data = scaler.fit_transform(filtered_diff_data)

    # data_frame = pd.DataFrame(scaled_diff_selected_data)
    data_frame = pd.DataFrame(scaled_selected_data[indexes, :])

    # Sort variables according to correlation
    correlations_investigated_temperature = data_frame.corr(method='spearman')[index_investigated_temperature]
    sorted_correlations = sorted([(a, b) for (a, b) in zip(variable_names, correlations_investigated_temperature.tolist())], key=lambda t2: t2[1], reverse=True)

    for sorted_correlation in sorted_correlations:
        print(f"{sorted_correlation[0]} => {sorted_correlation[1]}")

    # Draw graph linking data between columns and investigated temperature
    consumption_interval_W = 50
    for i in range(0, len(variable_names)):
        for j in range(0, 210, consumption_interval_W):
            cons_min = j
            cons_max = j + consumption_interval_W

            server_consumption_index = variable_names.index("ecotype_43_consumption")
            filtered_diff_data_cons_t2 = [(idx, row)
                                          for idx, row in zip(range(0, len(selected_data.tolist())), selected_data.tolist())
                                          if abs(row[server_consumption_index]) >= cons_min and abs(row[server_consumption_index]) <= cons_max]
            filtered_diff_data_cons = [t2[1] for t2 in filtered_diff_data_cons_t2]
            indexes_cons = [t2[0] for t2 in filtered_diff_data_cons_t2]

            investigated_values = selected_data[indexes_cons, index_investigated_temperature]
            investigated_variable_name = variable_names[index_investigated_temperature]

            # index_investigated_temperature2 = variable_names.index("ecotype_44_temperature")
            index_investigated_temperature2 = variable_names.index("ecotype_44_temperature")
            colors = selected_data[indexes_cons, index_investigated_temperature2]

            column_values = selected_data[indexes_cons, i]
            variable_name = variable_names[i]

            plt.scatter(column_values, investigated_values, c=colors, s=0.5)

            plt.legend()

            plt.title(f'{investigated_variable_name} vs {variable_name} cons in [{cons_min}, {cons_max}]')
            plt.xlabel(f'{variable_name}')
            plt.ylabel(f'{investigated_variable_name}')
            plt.colorbar()

            plt.savefig(f"{figure_folder}/{variable_name}_{cons_min}_correlation.pdf")
            plt.clf()

    return True
