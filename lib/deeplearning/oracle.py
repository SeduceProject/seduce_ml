from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape


def build_oracle(nb_servers):
    model = Sequential([
        Dense(256, input_shape=(nb_servers,)),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(output_dim=48, activation='linear')
        # Activation('softmax'),
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
