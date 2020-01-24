from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

NB_NEIGHBOURS = 5


class DebugOracle(object):

    def __init__(self, x_train, y_train, tss_train):
        self.x_train = x_train
        self.y_train = y_train
        self.tss_train = tss_train

        # Instantiate a Gaussian Process model
        param = [5 for x in range(0, x_train.shape[1])]

        kernel = RBF(param, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-2)
        self.gp.fit(self.x_train, self.y_train)

    def predict(self, x_input):
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, sigma = self.gp.predict(x_input, return_std=True)

        return y_pred[0], sigma
