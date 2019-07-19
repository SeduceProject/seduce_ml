from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
import keras.backend as K


def build_oracle(nb_servers):
    model = Sequential([
        Dense(53, input_shape=(48,)),
        Activation('tanh'),
        # Dense(53),
        # Activation('tanh'),
        # Dense(53),
        # Activation('tanh'),
        Dense(output_dim=48),
        Activation('linear'),
    ])

    model.compile(optimizer='adam',
                  loss='mse')

    return model


def train_oracle(oracle, data, epochs, batch_size):

    oracle.fit(data.get("x"),
               data.get("y"),
               epochs=epochs,
               batch_size=batch_size)

    pass


def run_oracle(oracle, input):
    pass
