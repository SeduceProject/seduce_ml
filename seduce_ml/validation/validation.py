import pandas
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from seduce_ml.data.scaling import *
from sklearn.linear_model import LinearRegression


def validate_seduce_ml(consumption_data, server_id, use_scaler, scaler_path=None, tmp_figures_folder=None, figure_label="", oracle_object=None):

    if oracle_object is None:
        raise Exception("Please provide an oracle object")

    x = consumption_data.get("x")
    y = consumption_data.get("y")
    unscaled_x = consumption_data.get("unscaled_x")
    unscaled_y = consumption_data.get("unscaled_y")
    tss = consumption_data.get("tss")

    scaler = oracle_object.scaler
    metadata = oracle_object.metadata

    [server_idx] = [idx for idx, e in enumerate(metadata.get("output_variables")) if e.get("output_of", None) == server_id]

    if scaler is None:
        if scaler_path is None:
            raise Exception("Please provide one of those arguments: ['scaler', 'scaler_path']")
        scaler = joblib.load(scaler_path)

    start_step = 0
    end_step = len(y)

    plot_data = []

    for idx, _ in enumerate(y):
        server_power_consumption = np.array(unscaled_x[idx])
        expected_value = y[idx]
        expected_temp = unscaled_y[idx]

        predicted_temp = oracle_object.predict(server_power_consumption)
        raw_result = rescale_output(predicted_temp, scaler, metadata.get("variables"))

        mse = ((predicted_temp - expected_temp) ** 2)

        plot_data += [{
            "x": idx,
            "y_actual": expected_value,
            "y_pred": raw_result,
            "mse_mean": mse.mean(axis=0),
            "mse_raw": mse,
            "rmse_mean": np.sqrt(mse).mean(axis=0),
            "rmse_raw": np.sqrt(mse),
            "temp_actual": expected_temp,
            "temp_pred": predicted_temp,
            "server_power_consumption": server_power_consumption,
        }]

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 95)].mean()

    if tmp_figures_folder is not None:

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][0][server_idx] for d in sorted_plot_data]

        dts = [dt.datetime(*[int(x)
                             for x in [a.split("-") + b.split(":")
                                       for (a, b) in [ts.replace("Z", "").split("T")]][0]])
               for ts in tss]

        sorted_dts = sorted(dts)

        # if oracle_object.learning_method == "ltsm":
        #     sorted_dts = sorted_dts[-len(y1_data):]

        ax.plot(sorted_dts, y1_data, color='blue', label='actual temp.', linewidth=0.5)
        ax.plot(sorted_dts, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)
        plt.legend()

        ax2 = ax.twinx()
        ax2.legend(loc=0)
        plt.legend()

        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of %s (deg. C)' % (server_id))

        plt.savefig((f"{tmp_figures_folder}/{figure_label}.pdf")
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

    return rmse, rmse_perc


def get_type_var(var):
    if "server_consumption" in var:
        return "consumption"
    if "server_temperature" in var:
        return "temperature"
    return "unknown"


def get_server_var(var):
    if "server_consumption" in var:
        return var.get("server_consumption")
    if "server_temperature" in var:
        return var.get("server_temperature")
    return "unknown"


def enrich_variables(vars):
    result = [
        dict([(k, v) for k, v in var.items()] +
             [("type", get_type_var(var))] +
             [("server", get_server_var(var))] +
             [("idx", idx)] +
             [("shift_count", var.get("shift_count", 0))])
        for (idx, var) in enumerate(vars)
        if not var.get("output", False)
    ]

    return result


def extract_substitutions(variables):
    """" Extract the substitution of variables that should be made between "row n" => "row n-1"
    """

    result = {}

    additional_variables_with_types = enrich_variables(variables)

    for var in additional_variables_with_types:
        similar_variables = [var2
                             for var2 in additional_variables_with_types
                             if var.get("type") == var2.get("type")
                             and var.get("server") == var2.get("server")
                             and var.get("shift_count") == var2.get("shift_count") -1]

        if len(similar_variables) > 0:
            similar_variable = similar_variables[-1]
            result[similar_variable.get("name")] = var.get("name")

    return result


def collect_substitutions_values(row, variables, substitutions):
    """" Collects the values of substitions from the given row
    """

    result = {}

    additional_variables_with_types = enrich_variables(variables)

    for substitued_variable_name, substituing_variable_name in substitutions.items():
        substituing_variables = [var
                             for var in additional_variables_with_types
                             if substituing_variable_name == var.get("name")]

        if len(substituing_variables) > 0:
            substituing_variable = substituing_variables[-1]
            substituing_value = row[substituing_variable["idx"]]
            result[substitued_variable_name] = substituing_value

    return result


def replace_row_variables(row, variables, new_values):
    """" Replace given variables in the specified row
    """

    if len(row.shape) > 1:
        raise Exception("row should be an 1D array")

    result = row.copy()

    input_variables = [var for var in variables if not var.get("output", False)]

    for column_idx, value in enumerate(row):
        column_var_name = input_variables[column_idx]["name"]
        if column_var_name in new_values:
            [new_column_value] = [value for (var_name, value) in new_values.items() if var_name == column_var_name]
            result[column_idx] = new_column_value

    return result


def shift_data_ltsm_non_scaled(x, variables, initial_substitutions, skip_types=[]):
    """
    """

    if len(x.shape) != 2:
        raise Exception("Expect the given 'x' array to be a 2D array")

    new_rows = []

    substitutions = extract_substitutions(variables)

    variables_with_metadata = enrich_variables(variables)
    input_variables = [var for var in variables_with_metadata if not var.get("output", False)]

    for idx, row in enumerate(x):
        # corresponding_input_var = input_variables[idx]
        #
        # if corresponding_input_var["type"] in skip_types:
        #     continue
        #
        # if idx == 0:
        #     substituing_values = collect_substitutions_values(row, variables, substitutions)
        #     new_row = replace_row_variables(row, variables, substituing_values)
        #     new_row = replace_row_variables(new_row, variables, initial_substitutions)
        #
        #     new_rows += [new_row]

        new_row = row.copy()

        new_rows += [new_row]

    return np.array(new_rows[:-1])


def shift_data_ltsm(x, variables, scaler, initial_substitution, skip_types=[]):
    unscaled_input = unscale_input(x, scaler, variables)

    unscaled_result = shift_data_ltsm_non_scaled(unscaled_input, variables, initial_substitution, skip_types=skip_types)

    return rescale_input(unscaled_result, scaler, variables)


def shift_data_non_scaled(x, variables, initial_substitutions, skip_types=[]):
    """
    """

    if len(x.shape) != 2:
        raise Exception("Expect the given 'x' array to be a 2D array")

    new_rows = []

    substitutions = extract_substitutions(variables)

    variables_with_types = enrich_variables(variables)
    input_variables = [var for var in variables_with_types if not var.get("output", False)]

    for idx, row in enumerate(x):

        corresponding_input_var = input_variables[idx]

        if corresponding_input_var["type"] in skip_types:
            substituing_values = collect_substitutions_values(row, variables, substitutions)
            new_row = replace_row_variables(row, variables, substituing_values)
        else:
            new_row = row
        new_row2 = replace_row_variables(new_row, variables, initial_substitutions)

        new_rows += [new_row2]

    return np.array(new_rows)


def shift_data(x, variables, scaler, initial_substitutions, skip_types=[]):
    unscaled_input = unscale_input(x, scaler, variables)

    unscaled_result = shift_data_non_scaled(unscaled_input, variables, initial_substitutions, skip_types=skip_types)

    return rescale_input(unscaled_result, scaler, variables)


LINEAR_REGRESSORS = {}


def compute_h_value(diff_y, current_cycle_counter, time_periods_without_real_temp, metadata):
    data = pandas.read_csv('probes.csv')

    h_values = []

    for idx, column_name in enumerate(metadata.get("output")):
        if column_name not in LINEAR_REGRESSORS:
            LINEAR_REGRESSORS[column_name] = {}
            for cycle_counter in range(0, time_periods_without_real_temp):
                data_current_cycle_counter = data.loc[data["current_cyle_counter"] == current_cycle_counter]
                X = data_current_cycle_counter[f"{column_name}"].values.reshape(-1, 1)
                Y = data_current_cycle_counter[f"h_{column_name}"].values.reshape(-1, 1)
                linear_regressor = LinearRegression()
                linear_regressor.fit(X, Y)
                LINEAR_REGRESSORS[column_name][cycle_counter] = linear_regressor

        Y_pred = LINEAR_REGRESSORS[column_name][current_cycle_counter].predict([[diff_y[0][idx]]])
        h_values += [Y_pred]

    return np.array(h_values).reshape(diff_y.shape)


def evaluate_prediction_power(consumption_data, server_id, tmp_figures_folder=None, figure_label="", produce_figure=True, oracle_object=None):

    if oracle_object is None:
        raise Exception("Please provide an oracle_object")

    metadata = oracle_object.metadata

    x = consumption_data.get("x")
    y = consumption_data.get("y")
    unscaled_x = consumption_data.get("unscaled_x")
    unscaled_y = consumption_data.get("unscaled_y")
    tss = consumption_data.get("tss")

    [server_idx] = [idx for idx, e in enumerate(metadata.get("output_variables")) if e.get("output_of", None) == server_id]

    start_step = 0
    end_step = len(y)

    plot_data = []

    time_periods_without_real_temp = 10

    resync_counter = 0
    current_cycle_counter = 0
    variables_that_travels = [var for var in metadata.get("variables") if var.get("become") is not None]

    probes = []

    for (idx, (e, ts)) in enumerate(zip(y, tss)):

        servers_consumptions = unscaled_x[idx]
        expected_temp = unscaled_y[idx]
        expected_value = y[idx]

        ts_to_date = dt.datetime(*[int(x)
                                   for x in [a.split("-") + b.split(":")
                                             for (a, b) in [ts.replace("Z", "").split("T")]][0]])

        predicted_temp = oracle_object.predict(servers_consumptions)

        if idx > 0 and (idx % time_periods_without_real_temp) != 0:
            resync = False
        else:
            resync = True

        if resync:
            resync_counter += 1

        current_cycle_counter = 0 if resync else current_cycle_counter + 1

        if current_cycle_counter == 0:
            h = np.zeros(len(metadata.get("output")))
            h_bis = h

        servers_consumptions_copy = servers_consumptions.copy()
        if not resync and resync_counter > 0:
            # Reuse previous variables
            _y = result_reusing_past_pred - h
            for var in variables_that_travels:
                substituting_output_var_idx = metadata.get("output").index(var.get("name"))
                substituting_input_var_idx = metadata.get("input").index(var.get("become"))
                servers_consumptions_copy[substituting_input_var_idx] = _y[0, substituting_output_var_idx]

            predicted_temp_reusing_past_pred = oracle_object.predict(servers_consumptions_copy)

            # rescaled_temps = rescale_output(predicted_temp_reusing_past_pred, oracle_object.scaler, oracle_object.metadata.get("variables"))
            # h_bis = compute_h_value(rescaled_temps, current_cycle_counter, time_periods_without_real_temp, metadata)
            # h = h_bis
        else:
            predicted_temp_reusing_past_pred = predicted_temp

        if current_cycle_counter == 0:
            raw_prediction = oracle_object.predict(servers_consumptions)

            # h = raw_prediction - expected_temp

            # print(f"{resync_counter} => mean: {h[0][server_idx]}")

        if idx > 0:
            raw_prediction = oracle_object.predict(servers_consumptions)

            # local_h = predicted_temp_reusing_past_pred - predicted_temp
            # h = local_h
            local_h = raw_prediction - expected_temp

            diff_x = x[idx] - x[idx-1]
            diff_y = (predicted_temp - unscale_output(np.array([y[idx - 1]]), oracle_object.scaler, oracle_object.metadata.get("variables"))).flatten()

            probes += [np.hstack((y[idx], diff_y, diff_x, local_h.flatten(), [current_cycle_counter])).ravel()]


        result_reusing_past_pred = predicted_temp_reusing_past_pred

        mse = ((predicted_temp_reusing_past_pred - expected_temp) ** 2)
        # h = np.zeros(len(metadata.get("output")))

        plot_data += [{
            "x": idx,
            "y_actual": expected_value,
            "y_pred": result_reusing_past_pred,
            "mse_mean": mse.mean(axis=0),
            "mse_raw": mse,
            "rmse_mean": np.sqrt(mse).mean(axis=0),
            "rmse_raw": np.sqrt(mse),
            "temp_actual": expected_temp,
            "temp_pred": predicted_temp - h_bis,
            "temp_pred_reusing_past_pred": predicted_temp_reusing_past_pred - h_bis,
            "resync": resync,
            "ts_to_date": ts_to_date
        }]

    # Export probes
    variables_names = [o.get("name") for o in metadata.get("output_variables")]
    variables_names += ["diff_" + o.get("name") for o in metadata.get("output_variables")]
    variables_names += ["diff_" + o.get("name") for o in metadata.get("input_variables")]
    variables_names += ["h_" + o.get("name") for o in metadata.get("output_variables")]
    variables_names += ["current_cyle_counter"]

    series_as_dataframe = pandas.DataFrame(probes, columns=variables_names)
    series_as_dataframe.to_csv(f"probes.csv",
                               columns=variables_names)

    # RMSE
    flatten_rmse = np.array([d["rmse_raw"] for d in plot_data]).flatten()
    rmse = flatten_rmse.mean()
    rmse_perc = flatten_rmse[flatten_rmse > np.percentile(flatten_rmse, 95)].mean()

    if tmp_figures_folder is not None and produce_figure:

        sorted_plot_data = sorted(plot_data, key=lambda d: d["x"])[start_step:end_step]

        fig = plt.figure()
        ax = plt.axes()

        y1_data = [d["temp_actual"][server_idx] for d in sorted_plot_data]
        y2_data = [d["temp_pred"][0][server_idx] for d in sorted_plot_data]
        y3_data = [d["temp_pred_reusing_past_pred"][0][server_idx] for d in sorted_plot_data]
        # y5_data = [d["temp_pred_reusing_past_pred2"][server_idx] for d in sorted_plot_data]

        synced_sorted_dts = [d["ts_to_date"] for d in sorted_plot_data if d["resync"]]
        y4_data = [d["temp_pred_reusing_past_pred"][0][server_idx] for d in sorted_plot_data if d["resync"]]

        dts = [d["ts_to_date"] for d in sorted_plot_data]

        sorted_dts = sorted(dts)
        sorted_dts = sorted_dts[0:len(sorted_plot_data)]

        ax.plot(sorted_dts, y1_data, color='blue', label='actual temp.', linewidth=0.5)
        ax.plot(sorted_dts, y2_data, color='red', label='predicted temp.', alpha=0.5, linewidth=0.5)
        ax.plot(sorted_dts, y3_data, color='green', label='predicted temp. (reusing past pred.)', alpha=0.5, linewidth=0.8)
        # ax.plot(sorted_dts, y5_data, color='black', label='predicted temp. (reusing past pred.)', alpha=0.5, linewidth=0.8)
        ax.scatter(synced_sorted_dts, y4_data, color='orange', marker='x', label='sync', alpha=0.5, linewidth=0.5)

        for i, (a, b) in enumerate(zip(synced_sorted_dts, y4_data)):
            ax.annotate(f"{i+1}", (a, b), fontsize=5)

        plt.legend()

        ax2 = ax.twinx()
        ax2.legend(loc=0)
        plt.legend()

        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()

        plt.xlabel('Time (hour)')
        plt.ylabel('Back temperature of %s (deg. C)' % (server_id))

        plt.savefig((f"{tmp_figures_folder}/{figure_label}_prediction_power.pdf")
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

        differential_dataframe = pandas.DataFrame(y1_data)\
            .diff().rename(columns={0: "diff"}).abs()\
            .rolling(5, win_type='triang').sum()\
            .assign(y1_data=y1_data)\
            .assign(y2_data=y2_data)

        top_095_quantile = differential_dataframe.quantile(0.80)
        filtered_dataframe = differential_dataframe.query(f"diff >= {top_095_quantile['diff']}")

        # ax1.xcorr(y1_data, filtered_dataframey2_data, usevlines=True, maxlags=50, normed=True, lw=2)
        ax1.xcorr(["y1_data"], filtered_dataframe["y1_data"], usevlines=True, maxlags=min(filtered_dataframe.size, 10), normed=True, lw=2)
        ax1.grid(True)

        # ax2.xcorr(y2_data, y3_data, usevlines=True, maxlags=50, normed=True, lw=2)
        # ax2.grid(True)

        plt.savefig((f"{tmp_figures_folder}/{figure_label}_auto_correlation_and_cross_correlation.pdf")
                    .replace(":", " ")
                    .replace("'", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "_")
                    )

    return rmse, rmse_perc
