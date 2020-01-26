from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from seduce_ml.oracle.oracle import Oracle


def flatten(ll):
    return [lj for li in ll for lj in li]


def build_oracle(nb_inputs, nb_outputs, hidden_layers_count=1, neurons_per_hidden_layer=53, activation_function="tanh"):
    model = Sequential()
    model.add(LSTM(4, activation=activation_function, input_shape=(neurons_per_hidden_layer, nb_inputs)))
    model.add(Dense(nb_outputs))

    model.compile(optimizer='adam',
                  loss='mse')

    return model


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def train_oracle(oracle, data, epochs, batch_size):
    _, n_steps, n_features = oracle.input_shape
    data_size, _ = data.get("x").shape

    x = data.get("x").reshape(data_size, n_features)
    y = data.get("y").reshape(data_size, 1)

    dataset = hstack((x, y))

    X, y = split_sequences(dataset, n_steps)

    oracle.fit(X,
               y,
               epochs=epochs,
               batch_size=batch_size)


class LstmOracle(Oracle):

    def __init__(self, oracle, scaler, metadata):
        Oracle.__init__(self, oracle, scaler, metadata, "ltsm")
