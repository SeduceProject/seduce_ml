from keras.models import Sequential
from keras.layers import Dense, Activation
from seduce_ml.oracle.oracle import Oracle
from seduce_ml.data.scaling import *
import copy


def build_oracle(nb_inputs, nb_outputs, hidden_layers_count=1, neurons_per_hidden_layer=53, activation_function="tanh"):
    model = Sequential(
        [
            Dense(neurons_per_hidden_layer, input_shape=(nb_inputs,)),
            Activation(activation_function),
        ]
        +
        flatten([
            [Dense(neurons_per_hidden_layer),
            Activation(activation_function)]
            for i in range(1, hidden_layers_count - 1)
        ])
        +
        [
            Dense(output_dim=nb_outputs),
            # Activation('linear'),
    ])

    model.compile(optimizer='adam',
                  loss='mse')

    return model


def train_oracle(oracle, data, epochs, batch_size):
    # oracle.fit(data.get("x"),
               # data.get("y"),
               # epochs=epochs,
               # batch_size=batch_size)

    from keras.callbacks import EarlyStopping
    overfitCallback = EarlyStopping(monitor='loss', min_delta=0.00001, patience=200)
    oracle.fit(data.get("x"),
               data.get("y"),
               epochs=100000000,
               # epochs=10,
               callbacks=[overfitCallback],
               batch_size=batch_size)

    oracle.compile(optimizer='rmsprop',
                   loss='mse')


class NeuralNetworkOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def train(self, data, params):
        self.data = data

        nb_layers = params.get("configuration").get("neural").get("layers_count")
        neurons_per_layers = params.get("configuration").get("neural").get("neurons_per_layer")
        activation_function = params.get("configuration").get("neural").get("activation_function")

        oracle = build_oracle(nb_inputs=len(self.metadata.get("input_variables")),
                              nb_outputs=len(self.metadata.get("output_variables")),
                              hidden_layers_count=nb_layers,
                              neurons_per_hidden_layer=neurons_per_layers,
                              activation_function=activation_function)

        x_train = data.scaled_train_df[data.metadata.get("input")].to_numpy()
        y_train = data.scaled_train_df[data.metadata.get("output")].to_numpy()
        x_test = data.scaled_test_df[data.metadata.get("input")].to_numpy()
        y_test = data.scaled_test_df[data.metadata.get("output")].to_numpy()

        train_oracle(oracle,
                     {
                         "x": x_train,
                         "y": y_train
                     },
                     params.get("epoch"),
                     data.scaled_train_df.shape[0])

        self.oracle = oracle

        # Evaluate the neural network
        score = oracle.evaluate(x_test, y_test, batch_size=params.get("batch_size"))

        self.state = "TRAINED"

        return score

    def predict(self, unscaled_input_values):
        # Transform 'unscaled_input_values' -> 'scaled_input_values'
        # Do a prediction from 'scaled_input_values' -> 'scaled_output_values'
        # Transform 'scaled_output_values' -> 'unscaled_output_values'
        # Return 'unscaled_ouput_values'
        scaled_input_values = rescale_input(unscaled_input_values.reshape(1, len(self.metadata.get("input"))),
                                            self.scaler,
                                            self.metadata.get("variables"))

        raw_result = self.oracle.predict(scaled_input_values)
        rescaled_result = unscale_output(raw_result, self.scaler, self.metadata.get("variables"))
        return rescaled_result.reshape(1, len(self.metadata.get("output")))
