from ..algo import Algo
import numpy as np
import pandas as pd
from .. import tools
from olmar import OLMAR

class OLMAR_BAH(Algo):
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, b0 = np.ones(6)/6.0, weight =None, MAX_WINDOW = 10, eps=1.01):
        super(OLMAR_BAH, self).__init__(min_history=MAX_WINDOW)

        # input check
        if MAX_WINDOW <= 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')
        self.max_window = MAX_WINDOW
        self.OLMAR_TRADERS = []
        for w in range(3,self.max_window):
            new_olmar = OLMAR(b0 = b0, weight = 1.0/(MAX_WINDOW-2),window = w, eps = eps)
            self.OLMAR_TRADERS.append(new_olmar)
        self.eps = eps
        self.b0 = b0

    def init_weights(self, m):
        if self.b0 !=None:
            return self.b0
        return np.ones(m) / m

    def step(self, x, last_b, history, **kwargs):
        for trader in self.OLMAR_TRADERS:
            trader.weight = trader.weight * np.dot(trader.last_b,x)
            x_pred = trader.predict(x, history.iloc[-trader.window:])
            b = trader.update(trader.last_b, x_pred, self.eps)
        sum_weights = np.sum([trader.weight for trader in self.OLMAR_TRADERS])
        for trader in self.OLMAR_TRADERS:
            trader.weight = trader.weight/sum_weights
        b = np.zeros(len(x))
        for trader in self.OLMAR_TRADERS:
            b += trader.last_b*trader.weight
        return b
