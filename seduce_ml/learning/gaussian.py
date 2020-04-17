from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from seduce_ml.oracle.oracle import Oracle
from seduce_ml.data.scaling import *

NB_NEIGHBOURS = 5
SUBSAMPLES = 2000


class GaussianProcessOracle(Oracle):

    def __init__(self, scaler, metadata, params):
        Oracle.__init__(self, scaler, metadata, params)

    def train(self, data, params):
        self.data = data

        x_train = self.data.scaled_train_df[self.data.metadata.get("input")].to_numpy()
        y_train = self.data.scaled_train_df[self.data.metadata.get("output")].to_numpy()

        # x_train = x_train[0: SUBSAMPLES]
        # y_train = y_train[0: SUBSAMPLES]

        # Instantiate a Gaussian Process model
        # kernel = RBF(param, (1e-2, 1e2))
        # kernel = DotProduct() + WhiteKernel()
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-2)
        self.gp.fit(x_train, y_train)

        self.state = "TRAINED"

    def predict(self, x_input):
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        rescaled_x_input = rescale_input(x_input.reshape(1, len(self.metadata.get("input"))), self.scaler, self.metadata.get("variables"))

        y_pred, sigma = self.gp.predict(rescaled_x_input, return_std=True)
        rescaled_result = unscale_output(y_pred, self.scaler, self.metadata.get("variables"))
        return rescaled_result.reshape(1, len(self.metadata.get("output")))
