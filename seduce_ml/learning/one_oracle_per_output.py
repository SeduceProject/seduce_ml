from seduce_ml.data.scaling import *


class SimpleOneOraclePerOutput():

    def __init__(self, oracles):
        self.oracles = oracles

    def predict(self, unscaled_input_values):
        result = []
        for local_oracle in self.oracles:
            local_oracle_input_columns = local_oracle.metadata.get("input")
            input_array = unscaled_input_values[local_oracle_input_columns].to_numpy()
            result += [local_oracle.predict(input_array)]
        return np.array(result).reshape(1, len(self.oracles))
