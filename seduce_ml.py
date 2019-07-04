import sys
from lib.deeplearning.oracle import build_oracle, train_oracle
from lib.data.seduce_data_loader import simulate_consumption_function, generate_fake_consumption_data, generate_real_consumption_data, average_temperature_aggregated_by_minute
from lib.data.seduce_data_loader import NORMALIZATION_COOLING, NORMALIZATION_SERVER, normalize_cooling, denormalize_temperature
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Dense
import keras

import math

import numpy as np
from keras.models import load_model
import time;


EPOCH_COUNT = 1000
BATCH_SIZE = 1000


def sort_tasks_scheduler(loads):
    return sorted(loads)


def sort_tasks_scheduler(loads):
    return sorted(loads)


if __name__ == "__main__":
    print("Hello from seduce ML")

    nb_servers = 48
    nb_run = 50

    train = True

    if train:


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

            # Build the neural network
            oracle = build_oracle(nb_servers)

            # Train the neural network

            # ##################
            # # USE FAKE DATA
            # x_train, y_train = generate_fake_consumption_data(nb_servers, 2000)
            # x_test, y_test = generate_fake_consumption_data(nb_servers, 100)

            ###################
            # # USE REAL DATA
            x, y, tss = generate_real_consumption_data()
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

                difference = (result - expected_value) * (result - expected_value)
                sum_squared_difference += difference
            std = math.sqrt(sum_squared_difference / len(y))
            print("standard deviation: %s" % std)

            prediction_failed = 0
            differences = []
            signed_differences = []
            for idx in range(0, len(y)):
                test_input = np.array([x[idx]])
                expected_value = denormalize_temperature(y[idx])

                result = denormalize_temperature(oracle.predict(test_input)[0][0])

                # difference = math.sqrt((result - expected_value) * (result - expected_value))
                difference = math.sqrt((result - expected_value) * (result - expected_value))
                signed_differences += [(result - expected_value)]

                # if result > 0.30:
                differences += [difference]

                if difference >= 1.0:
                    print("%s: expected:%s --> predicted:%s (%s)" % (np.mean(test_input) * NORMALIZATION_SERVER, expected_value, result, tss[idx]))
                    prediction_failed += 1
            # std = math.sqrt(sum_squared_difference / len(y))
            print("%s / %s prediction were too far from real data" % (prediction_failed, len(y)))

            average_difference = normalize_cooling(float(np.mean(differences)))
            print("average difference: %s" % (average_difference))

            average_signed_difference = normalize_cooling(float(np.mean(signed_differences)))
            print("average signed difference: %s" % (average_signed_difference))

            differences_after_correction = []

            oracle.add(keras.layers.Lambda(lambda x, k: x - k, arguments={"k": average_signed_difference / NORMALIZATION_COOLING}))

            oracle.compile(optimizer='rmsprop',
                           loss='mse')

            adjusted_prediction_failed = 0
            for idx in range(0, len(y)):
                test_input = np.array([x[idx]])
                expected_value = denormalize_temperature(y[idx])

                result = denormalize_temperature(oracle.predict(test_input)[0][0])

                # difference = math.sqrt((result - expected_value) * (result - expected_value))
                difference = math.sqrt((result - expected_value) * (result - expected_value))

                differences_after_correction += [difference]

                if difference > 1.0:
                    print("%s: expected:%s --> adjusted_predicted:%s (%s)" % (np.mean(test_input) * NORMALIZATION_SERVER, expected_value, result, tss[idx]))
                    adjusted_prediction_failed += 1
            # std = math.sqrt(sum_squared_difference / len(y))
            print("%s / %s adjusted prediction were too far from real data" % (adjusted_prediction_failed, len(y)))

            average_difference_after_correction = normalize_cooling(float(np.mean(differences_after_correction)))
            print("average difference after correction: %s" % (average_difference_after_correction))

            averages_differences += [average_difference]
            averages_differences_after_correction += [average_difference_after_correction]
            scores += [score]

            if best_score is None or score < best_score:
                best_score = score

            if best_difference is None or average_difference < best_difference:
                best_difference = average_difference
                # best_oracle = oracle

            if best_difference_after_correction is None or average_difference_after_correction < best_difference_after_correction:
                best_signed_difference = average_difference_after_correction
                best_oracle = oracle

        mean_difference = float(np.mean(averages_differences))
        mean_score = float(np.mean(score))
        print("mean_difference: %s" % (mean_difference))
        print("best_difference: %s" % (best_difference))
        print("best_difference_after_correction: %s" % (best_signed_difference))
        print("mean_score: %s" % (mean_score))
        print("best_score: %s" % (best_score))

        oracle = best_oracle

    else:
        oracle = load_model('/Users/jonathan/seduceml3.h5')

    epoch = time.time()
    date_str = time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime(epoch))

    oracle.save("data/seduceml_%s.h5" % date_str)

    sys.exit(0)
