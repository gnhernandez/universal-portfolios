from ..algo import Algo
import numpy as np
import pandas as pd
from .. import tools


class OLMAR(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True
    EMA_FLAG = True
    EMA_ALPHA = 0.25

    def __init__(self, b0 =None, weight = None, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history=window)

        # input check
        if window <= 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps
        self.b0 = b0
        self.weight = weight
        self.last_b = b0



    def init_weights(self, m):
        if self.b0:
            return self.b0
        return np.ones(m) / m


    def step(self, x, last_b, history, **kwargs):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        self.last_b = b
        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        if not OLMAR.EMA_FLAG:
            return (history / x).mean()
        else:
            ema_df = (pd.ewma(history,com = (1.0/OLMAR.EMA_ALPHA) -1)[-1:]/x).transpose()
            return pd.Series(ema_df.values.flatten(), ema_df.index)


    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        denom = np.linalg.norm(x - x_mean)**2
        if denom == 0:
            lam = 0
        else:
            lam = max(0., (eps - np.dot(b, x)) / denom)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)
        b = b + lam * (x - x_mean)

        allocation = self.simplex_projection(b)
        self.last_b = allocation
        return allocation

    def simplex_projection(self,v, b=1):
        v = np.asarray(v)
        p = len(v)
        # Sort v into u in descending order
        v = (v > 0) * v
        u = np.sort(v)[::-1]
        sv = np.cumsum(u)
        rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
        theta = np.max([0, (sv[rho] - b) / (rho+1)])
        w = (v - theta)
        w[w<0] = 0
        return w
