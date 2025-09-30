import math
import numpy as np


class DataGenerator:

    def __init__(self):
        pass

    def gen_circle(self, n_samples=1000, dist='1d', df=3):
        data = np.zeros((n_samples, 2))
        theta = 2 * math.pi * np.random.rand(n_samples)
        if dist == 'chisq':
            if not (isinstance(df, int) and df > 2):
                df = 3
            r = np.random.chisquare(df, size=n_samples) / df
        else:
            r = np.ones(n_samples)
        data[:, 0] = r * np.cos(theta)
        data[:, 1] = r * np.sin(theta)
        self.data = data
        return data

    def gen_sine(self, n_samples=1000, std=0):
        data = np.zeros((n_samples, 2))
        data[:, 0] = 2 * math.pi * (np.random.rand(n_samples) - 0.5)
        data[:, 1] = np.sin(data[:, 0])
        if std > 0:
            data[:, 1] = data[:, 1] + np.random.normal(loc=0.0, scale=std, size=n_samples)
        self.data = data
        return data