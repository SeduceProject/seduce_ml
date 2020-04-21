import pandas
import os
import matplotlib.pyplot as plt
from seduce_ml.data.scaling import *


def evaluate_prediction_power(consumption_data, server_id, params=None, tmp_figures_folder=None, figure_label="", produce_figure=True, oracle_object=None, group_by=-1):

    if oracle_object is None:
        raise Exception("Please provide an oracle_object")

    metadata = oracle_object.metadata

    x = consumption_data.scaled_df[consumption_data.metadata.get("input")].to_numpy()
    y = consumption_data.scaled_df[consumption_data.metadata.get("output")].to_numpy()
    unscaled_x = consumption_data.unscaled_df[consumption_data.metadata.get("input")].to_numpy()
    unscaled_y = consumption_data.unscaled_df[consumption_data.metadata.get("output")].to_numpy()
    tss = consumption_data.unscaled_df["timestamp"].to_numpy()

    [server_idx] = [idx for idx, e in enumerate(metadata.get("output_variables")) if e.get("output_of", None) == server_id]

    start_step = 0
    end_step = len(y)

    STEPS_TO_PREDICT = 4

    probes = []

    for i in range(STEPS_TO_PREDICT + 1):

        grouped_rows = [unscaled_x[idx: idx + i + 1]
                        for idx, e in enumerate(tss)
                        if unscaled_x[idx: idx + i + 1].shape[0] == i+1]

        grouped_rows_array = np.array(grouped_rows).reshape(len(grouped_rows), i + 1, x.shape[-1])

        predicted_temps = oracle_object.predict_all_nsteps_in_future(grouped_rows_array, nsteps=i)

        expected_temps = unscaled_y[i:]

        plot_data = [
            {
                "x": idx,
                "temp_expected": et,
                "temp_pred": pt,
                "ts": tss[idx]
            }
            for idx, (et, pt) in enumerate(zip(expected_temps, predicted_temps))
        ]

        # Export probes
        variables_names = [o.get("name") for o in metadata.get("output_variables")]
        variables_names += ["diff_" + o.get("name") for o in metadata.get("output_variables")]
        variables_names += ["diff_" + o.get("name") for o in metadata.get("input_variables")]
        variables_names += ["h_" + o.get("name") for o in metadata.get("output_variables")]
        variables_names += ["current_cyle_counter"]

        series_as_dataframe = pandas.DataFrame(probes, columns=variables_names)
        series_as_dataframe.to_csv(f"probes.csv",
                                   columns=variables_names)

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        y_expected_temperatures = [d["temp_expected"][server_idx] for d in sorted_plot_data]
        y_predicted_temperatures = [d["temp_pred"][server_idx] for d in sorted_plot_data]

        if tmp_figures_folder is not None and produce_figure:

            fig = plt.figure()
            ax = plt.axes()

            dts = [d["ts"] for d in sorted_plot_data]

            sorted_dts = sorted(dts)
            sorted_dts = sorted_dts[0:len(sorted_plot_data)]

            ax.plot(sorted_dts, y_expected_temperatures, color='blue', label='actual temp.', linewidth=0.5)
            ax.plot(sorted_dts, y_predicted_temperatures, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)

            plt.legend()

            ax2 = ax.twinx()
            ax2.legend(loc=0)
            plt.legend()

            ax.xaxis_date()

            # Make space for and rotate the x-axis tick labels
            fig.autofmt_xdate()

            plt.xlabel('Time (hour)')
            plt.ylabel('Back temperature of %s (deg. C)' % (server_id))
            plt.title(f'Prediction {i * oracle_object.params.get("seduce_ml").get("group_by") } minutes ahead')

            plt.savefig((f"{tmp_figures_folder}/{server_id}_{figure_label}_prediction_power_step={i}.pdf")
                        .replace(":", " ")
                        .replace("'", "")
                        .replace("{", "")
                        .replace("}", "")
                        .replace(" ", "_")
                        )

            # Plotting auto-correlation and cross-correlations
            fig, ax1 = plt.subplots(1, 1, sharex=True)

            # y1_pandas_diff = pandas.DataFrame()
            # ax1.acorr(y2_data, maxlags=9)

            differential_dataframe = pandas.DataFrame(y_expected_temperatures)\
                .diff().rename(columns={0: "diff"}).abs()\
                .rolling(5, win_type='triang').sum()\
                .assign(y1_data=y_expected_temperatures)\
                .assign(y2_data=y_predicted_temperatures)

            top_095_quantile = differential_dataframe.quantile(0.80)
            filtered_dataframe = differential_dataframe.query(f"diff >= {top_095_quantile['diff']}")

            # ax1.xcorr(y1_data, filtered_dataframey2_data, usevlines=True, maxlags=50, normed=True, lw=2)
            ax1.xcorr(filtered_dataframe["y1_data"], filtered_dataframe["y1_data"], usevlines=True, maxlags=min(filtered_dataframe.size, 10), normed=True, lw=2)
            ax1.grid(True)

            plt.savefig((f"{tmp_figures_folder}/{server_id}_{figure_label}_auto_correlation_and_cross_correlation.pdf")
                        .replace(":", " ")
                        .replace("'", "")
                        .replace("{", "")
                        .replace("}", "")
                        .replace(" ", "_")
                        )
            plt.close('all')

        if params and "output_csv" in params.get("seduce_ml") and produce_figure:
            output_csv_path = params.get("seduce_ml").get("output_csv")
            csv_file_path = f"{output_csv_path}/data_{ i }_steps.csv"

            if not os.path.exists(output_csv_path):
                os.makedirs(output_csv_path)

            df = pandas.DataFrame()
            if os.path.exists(csv_file_path):
                df = pandas.read_csv(csv_file_path, parse_dates=["timestamp"], index_col=0)

            server_id_underscore = server_id.replace("-", "_")

            df[f"timestamp"] = [d["ts"] for d in sorted_plot_data]
            df[f"{ params.get('seduce_ml').get('learning_method') }_{server_id_underscore}_predicted"] = [d["temp_pred"][server_idx] for d in sorted_plot_data]
            df[f"{ params.get('seduce_ml').get('learning_method') }_{server_id_underscore}_expected"] = [d["temp_expected"][server_idx] for d in sorted_plot_data]

            df.to_csv(csv_file_path)

    return True
