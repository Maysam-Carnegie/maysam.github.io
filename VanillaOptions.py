# Author(s): Maysam Khodayari

import time
import numpy as np
from matplotlib import pyplot as plt
from numpy import log,exp,sqrt
from scipy.stats import norm   # for Normal CDF

class PlainVanillaOption:
    def __init__(self, S0, K, r, sigma, T, f):
        self._S0 = S0        # initial stock price
        self._K = K          # strike price
        self._r = r          # risk-free rate
        self._sigma = sigma  # volatility
        self._T = T          # expiration time
        self._num_intervals =   int(input('Enter the number of time intervals up to maturity: '))
        self._f = f

    # Inner class used by the binomial tree model
    class _PriceNode:
        def __init__(self):
            self._stock_price = 0
            self._option_price = 0       

# EuropeanCallOption methods:
    def __str__(self):
        return ('EuropeanCallOption:\n'
                + '  S0: ' + str(self._S0)
                + '  K: ' + str(self._K)
                + '  r: ' + str(self._r)
                + '  sigma: ' + str(self._sigma)
                + '  T: ' + str(self._T))
    # Binomial Tree general parts
    def binomialPrice(self):
        # local variables used by this function
        num_intervals = self._num_intervals
        deltaT = self._T / num_intervals
        u = exp(self._sigma * deltaT ** 0.5)
        d = 1 / u
        a = exp(self._r * deltaT)
        p = (a - d) / (u - d)
        q = 1 - p
        K = self._K
        # abbreviated name
        ni = int(num_intervals)
        # fill tree with all 0.0s
        binom_tree = []
        for i in range(ni + 1):
            ith_row = [self._PriceNode()
                              for j in range(i+1)]
            binom_tree.append(ith_row)
        # fill tree with stock prices
        for i in range(ni + 1):
            for j in range(i + 1):
                binom_tree[i][j]._stock_price = (
                        self._S0 * u ** j * d ** (i-j))
        return self._f(binom_tree, p, q, a, K)

class EuropeanCallOption(PlainVanillaOption):
    def backWard(binom_tree, p, q, a, K):
        ni = len(binom_tree) - 1
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(binom_tree[ni][j]._stock_price - K,
                            0.0))
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = ((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a

        return binom_tree[0][0]._option_price   
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, EuropeanCallOption.backWard)
      

class EuropeanPutOption(PlainVanillaOption):
    def backWard(binom_tree, p, q, a, K):
        ni = len(binom_tree) - 1
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(K - binom_tree[ni][j]._stock_price,
                            0.0))
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = ((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a

        return binom_tree[0][0]._option_price   
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, EuropeanPutOption.backWard)
      

class AmericanCallOption(PlainVanillaOption):
    def backWard(binom_tree, p, q, a, K):
        ni = len(binom_tree) - 1
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(binom_tree[ni][j]._stock_price - K,
                            0.0))
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = max(((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a,\
                                                     binom_tree[i][j]._stock_price - K )

        return binom_tree[0][0]._option_price   
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, AmericanCallOption.backWard)
      
      
class AmericanPutOption(PlainVanillaOption):
    def backWard(binom_tree, p, q, a, K):
        ni = len(binom_tree) - 1
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(K - binom_tree[ni][j]._stock_price,
                            0.0))
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = max(((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a,\
                                                     K - binom_tree[i][j]._stock_price)

        return binom_tree[0][0]._option_price   
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, AmericanPutOption.backWard)
      

ec = EuropeanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ep = EuropeanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ac = AmericanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ap = AmericanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)

options = [ec, ep, ac, ap]

for option in options:
    time_b = time.time()
    option.binomialPrice()
    time_a = time.time()
    print(f'\nNon-vectorized version of {str(type(option))[17:-2]} took {time_a - time_b} seconds to run!')




