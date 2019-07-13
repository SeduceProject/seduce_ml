import sys
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import norm
from keras.models import load_model
from lib.deeplearning.oracle import build_oracle, train_oracle
from lib.data.seduce_data_loader import generate_real_consumption_data

EPOCH_COUNT = 10000
BATCH_SIZE = 1000


def sort_tasks_scheduler(loads):
    return sorted(loads)


def sort_tasks_scheduler(loads):
    return sorted(loads)


if __name__ == "__main__":
    print("Hello from seduce ML")

    nb_servers = 48
    nb_run = 1

    train = True

    if train:

        best_data = None
        best_plot_data = []
        scores = []
        averages_differences = []
        averages_differences_after_correction = []

        best_difference = None
        best_difference_after_correction = None
        best_score = None
        best_oracle = None

        for i in range(0, nb_run):
            # Display a message
            print("Run %s/%s" % (i, 50))

            plot_data = []

            # Build the neural network
            oracle = build_oracle(nb_servers)

            # Train the neural network

            # ##################
            # # USE FAKE DATA
            # x_train, y_train = generate_fake_consumption_data(nb_servers, 2000)
            # x_test, y_test = generate_fake_consumption_data(nb_servers, 100)

            ###################
            # # USE REAL DATA
            x, y, tss, data = generate_real_consumption_data()
            train_proportion = 0.5
            train_probe_size = int(len(x) * train_proportion)
            x_train, y_train = x[train_probe_size:], y[train_probe_size:]
            x_test, y_test = x[:train_probe_size], y[:train_probe_size]

            train_oracle(oracle,
                         {
                             "x": x_train,
                             "y": y_train
                         },
                         EPOCH_COUNT,
                         BATCH_SIZE)

            # Evaluate the neural network
            score = oracle.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
            print("score: %s" % (score))

            # Use the neural network

            sum_squared_difference = 0
            for idx in range(0, len(y)):
                test_input = np.array([x[idx]])
                expected_value = y[idx]

                result = oracle.predict(test_input)[0][0]

                difference = norm(expected_value - result)

                # difference = (result - expected_value) * (result - expected_value)
                sum_squared_difference += difference
            std = math.sqrt(sum_squared_difference / len(y))
            print("standard deviation: %s" % std)

            prediction_failed = 0
            differences = []
            signed_differences = []
            for idx in range(0, len(y)):
                test_input = np.array([x[idx]])
                expected_value = y[idx]
                result = oracle.predict(test_input)[0]

                difference = norm(expected_value - result)

                # if result > 0.30:
                differences += [difference]

                # if difference >= 1.0:
                #     print("%s: expected:%s --> predicted:%s (%s)" % (np.mean(test_input), expected_value, result, tss[idx]))
                #     prediction_failed += 1

                plot_data += [{
                    "x": idx,
                    "y_actual": expected_value,
                    "y_pred": result
                }]

            # std = math.sqrt(sum_squared_difference / len(y))
            print("%s / %s prediction were too far from real data" % (prediction_failed, len(y)))

            average_difference = float(np.mean(differences))
            print("average difference: %s" % (average_difference))

            differences_after_correction = []

            oracle.compile(optimizer='rmsprop',
                           loss='mse')

            averages_differences += [average_difference]

            # averages_differences_after_correction += [average_difference_after_correction]
            scores += [score]

            if best_score is None or score < best_score:
                best_score = score

            if best_difference is None or average_difference < best_difference:
                best_difference = average_difference
                best_oracle = oracle
                best_plot_data = plot_data
                best_data = data

        mean_difference = float(np.mean(averages_differences))
        mean_score = float(np.mean(score))
        print("mean_difference: %s" % (mean_difference))
        print("best_difference: %s" % (best_difference))
        # print("best_difference_after_correction: %s" % (best_signed_difference))
        print("mean_score: %s" % (mean_score))
        print("best_score: %s" % (best_score))

        oracle = best_oracle

    else:
        oracle = load_model('/Users/jonathan/seduceml3.h5')

    epoch = time.time()
    date_str = time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime(epoch))

    oracle.save("data/seduceml_%s.h5" % date_str)

    # Draw the comparison between actual data and prediction:
    # sorted_plot_data = sorted(best_plot_data, key=lambda d: d["y_actual"])

    start_step = int(0.66 * len(best_plot_data))
    end_step = int(0.75 * len(best_plot_data))

    sorted_plot_data = sorted(best_plot_data, key=lambda d: d["x"])[start_step:end_step]

    fig = plt.figure()
    ax = plt.axes()

    server_id = 1

    x_data = [d["x"] for d in sorted_plot_data]
    y1_data = [d["y_actual"][server_id] * (best_data["max_temperature"] - best_data["min_temperature"]) + best_data["min_temperature"] for d in sorted_plot_data]
    y2_data = [d["y_pred"][server_id] * (best_data["max_temperature"] - best_data["min_temperature"])+ best_data["min_temperature"] for d in sorted_plot_data]
    x_data = range(0, len(y1_data))

    ax.plot(x_data, y1_data, color='blue', label='actual max temp.')
    ax.plot(x_data, y2_data, color='red', label='predicted max temp.', alpha=0.5)

    plt.legend()

    plt.show()

    sys.exit(0)
