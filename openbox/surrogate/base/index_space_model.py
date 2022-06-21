import json
import numpy as np
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.logging_utils import get_logger


class IndexSpaceModel:
    def __init__(self, config_space: ConfigurationSpace,
                 budget=1500,
                 history='/tmp/indexsize.json', # {'table.column': index_size}
                 ):
        self.budget = budget
        self.config_space = config_space
        self.all_hps = config_space.get_hyperparameter_names()
        self.all_index_sizes = np.zeros(len(config_space.get_hyperparameter_names()))

        self.logger = get_logger(self.__class__.__name__)

        with open(history, 'r') as f:
            index_size = json.load(f)

        for i, hp in enumerate(self.all_hps):
            if not hp.startswith('index.'):
                continue
            index = hp[6:]
            self.all_index_sizes[i] = float(index_size[index])

        self.logger.debug('All index sizes: {}'.format(self.all_index_sizes))

    def train(self, X, cY, contexts: np.ndarray=None):
        return

    def p_feasible(self, X: np.ndarray):
        cost = np.dot(X, self.all_index_sizes)
        p = np.array((cost < self.budget), dtype=np.float64)
        p = np.reshape(p, (-1, 1))
        return p

