from keras.models import Sequential
from keras.layers import Dense, Activation


def flatten(ll):
    return [lj for li in ll for lj in li]


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

    oracle.fit(data.get("x"),
               data.get("y"),
               epochs=epochs,
               batch_size=batch_size)
