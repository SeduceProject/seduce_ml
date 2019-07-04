import pymc3 as pm
import random

with pm.Model():
    # captured = [689, 341, 386, 741, 982, 414, 845, 241, 180, 447, 880, 21, 583, 993, 812]
    captured = []
    for i in range(0, 100):
        captured += [random.uniform(0, 666)]


    num_tanks = pm.DiscreteUniform(
        "num_tanks",
        lower=max(captured),
        upper=2000
    )
    likelihood = pm.DiscreteUniform(
        "observed",
        lower=1,
        upper=num_tanks,
        observed=captured
    )
    posterior = pm.sample(10000, tune=1000)
    plot = pm.plot_posterior(posterior, credible_interval=0.95)
    import matplotlib.pyplot as plt

    # create fig1 (of type plt.figure)
    # create fig2

    plt.show()
