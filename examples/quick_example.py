# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, sp


# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objs': (y,)}


# Run
if __name__ == "__main__":
    opt = Optimizer(
        branin,
        space,
        max_runs=10,
        #surrogate_type='gp',
        surrogate_type='context_prf',
        time_limit_per_trial=30,
        task_id='quick_start',
       current_context= np.random.uniform(size=(1, 10)),
        context_pca_components=4,
    )

    history = opt.run()

    opt.max_iterations = 20

    opt.reset_context(np.random.uniform(size=(1, 10)))
    history = opt.run()

    print(history)

    history.plot_convergence(true_minimum=0.397887)
    plt.show()

    # install pyrfr to use get_importance()
    # print(history.get_importance())

    # history.visualize_jupyter()
