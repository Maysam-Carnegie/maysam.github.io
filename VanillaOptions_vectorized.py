# Author(s): Maysam Khodayari

import time
import numpy as np
from numpy import log,exp,sqrt, maximum, abs
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
        ni = num_intervals
        # fill tree with all 0.0s
        # binom_tree = []
        # for i in range(ni + 1):
        #     ith_row = [self._PriceNode()
        #                       for j in range(i+1)]
        #     binom_tree.append(ith_row)
        stock_prices = np.zeros((ni+1, ni+1))    # new vectorized method
               
        # fill tree with stock prices
        for i in range(ni + 1):
            for j in range(i+1):
                stock_prices[i][j] = self._S0 * u ** j * d ** (i-j)
            # for j in range(i + 1):
                # binom_tree[i][j]._stock_price = (
                #         self._S0 * u ** j * d ** (i-j))
        return self._f(stock_prices, p, q, a, K)

class EuropeanCallOption(PlainVanillaOption):
    def backWard(stock_prices, p, q, a, K):
        ni = len(stock_prices[-1]) - 1
        option_prices = np.zeros((ni+1, ni+1))   # new vectorized method
        # fill in terminal node option prices
        for j in range(ni + 1):
            option_prices[-1][j] = max(stock_prices[-1][j] - K, 0.0)
        for i in range(ni - 1, -1, -1):
            option_prices[i][0:i+1] = ((p * option_prices[i + 1][1:i+2]) +\
                                       (q * option_prices[i + 1][0:i+1])) / a
        return option_prices[0][0]
    
    def BSMPrice(self):        
        d1 = (log(self._S0 / self._K) + (self._r + (self._sigma**2) / 2 ) * (self._T)) / (self._sigma * self._T ** 0.5)
        d2 = d1 - self._sigma * self._T ** 0.5        
        C0 = (norm.cdf(d1) *  self._S0) - (norm.cdf(d2) * self._K * exp(-self._r * self._T))
        return C0
    
    def vcall(self, n):
        zi = np.random.randn(n)
        Si = self._S0 * exp((self._r - .5*self._sigma**2) * self._T \
                            + self._sigma * self._T**.5 * zi)
        Ci = exp(-self._r * self._T) * maximum(Si - self._K, 0)
        return Ci  
    
    def simPrice(self, pn = 0.005):
        n = 100_000
        v = EuropeanCallOption.vcall(self, n)        
        mn1 = np.mean(v)
        sums = 0
        for i in v:
            sums += (i - mn1)**2
        s1 = sqrt(sums/(n-1))
        while abs(s1/sqrt(n)) >= pn:
            n *= 2
            v = EuropeanCallOption.vcall(self, n)
            mn1 = np.mean(v)
            sums = 0
            for i in v:
                sums += (i - mn1)**2
            s1 = sqrt(sums/(n-1))            
        print("\nCall price:", np.mean(v), '\nfinal number of sample is: ', n)

    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, EuropeanCallOption.backWard)
      
class EuropeanPutOption(PlainVanillaOption):
    def backWard(stock_prices, p, q, a, K):
        ni = len(stock_prices[-1]) - 1
        option_prices = np.zeros((ni+1, ni+1))   # new vectorized method
        # fill in terminal node option prices
        for j in range(ni + 1):
            option_prices[-1][j] = max(K - stock_prices[-1][j], 0.0)
        for i in range(ni - 1, -1, -1):
            option_prices[i][0:i+1] = ((p * option_prices[i + 1][1:i+2]) +\
                                       (q * option_prices[i + 1][0:i+1])) / a
        return option_prices[0][0]
    
    def BSMPrice(self):        
        d1 = (log(self._S0 / self._K) + (self._r + (self._sigma**2) / 2 ) * (self._T)) / (self._sigma * self._T ** 0.5)
        d2 = d1 - self._sigma * self._T ** 0.5        
        C0 = (norm.cdf(-d2) * self._K * exp(-self._r * self._T)) - (norm.cdf(-d1) *  self._S0)
        return C0
    
    def vput(self, n):
        zi = np.random.randn(n)
        Si = self._S0 * exp((self._r - .5*self._sigma**2) * self._T \
                            + self._sigma * self._T**.5 * zi)
        Ci = exp(-self._r * self._T) * maximum(self._K - Si, 0)
        return Ci  
    
    def simPrice(self, pn = 0.005):
        n = 100_000
        v = EuropeanPutOption.vput(self, n)        
        mn1 = np.mean(v)
        sums = 0
        for i in v:
            sums += (i - mn1)**2
        s1 = sqrt(sums/(n-1))
        while abs(s1/sqrt(n)) >= pn:
            n *= 2
            v = EuropeanPutOption.vput(self, n)  
            mn1 = np.mean(v)
            sums = 0
            for i in v:
                sums += (i - mn1)**2
            s1 = sqrt(sums/(n-1))            
        print("\nPut price:", np.mean(v), '\nfinal number of sample is: ', n)
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, EuropeanPutOption.backWard)

class AmericanCallOption(PlainVanillaOption):
    def backWard(stock_prices, p, q, a, K):
        ni = len(stock_prices[-1]) - 1
        option_prices = np.zeros((ni+1, ni+1))   # new vectorized method
        # fill in terminal node option prices
        for j in range(ni + 1):
            option_prices[-1][j] = max(stock_prices[-1][j] - K, 0.0)
        for i in range(ni - 1, -1, -1):
            option_prices[i][0:i+1] = np.maximum(((p * option_prices[i + 1][1:i+2]) +\
                                       (q * option_prices[i + 1][0:i+1])) / a,\
                                          stock_prices[i][0:i+1] - K)
        return option_prices[0][0]
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, AmericanCallOption.backWard)


class AmericanPutOption(PlainVanillaOption):
    def backWard(stock_prices, p, q, a, K):
        ni = len(stock_prices[-1]) - 1
        option_prices = np.zeros((ni+1, ni+1))   # new vectorized method
        # fill in terminal node option prices
        for j in range(ni + 1):
            option_prices[-1][j] = max(K - stock_prices[-1][j], 0.0)
        for i in range(ni - 1, -1, -1):
            option_prices[i][0:i+1] = np.maximum(((p * option_prices[i + 1][1:i+2]) +\
                                       (q * option_prices[i + 1][0:i+1])) / a,\
                                          K - stock_prices[i][0:i+1])
        return option_prices[0][0]
    
    def __init__(self, S0, K, r, sigma, T):
      super().__init__(S0, K, r, sigma, T, AmericanPutOption.backWard)

#3.c
ec = EuropeanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ep = EuropeanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ac = AmericanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ap = AmericanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)

options = [ec, ep, ac, ap]

for option in options:
    time_b = time.time()
    option.binomialPrice()
    time_a = time.time()
    print(f'\nVectorized version of {str(type(option))[17:-2]} took {time_a - time_b} seconds to run!')

for option in options[0:2]:
    for pn in [0.005, 0.01, 0.001]:
        time_b = time.time()
        option.simPrice()
        time_a = time.time()
        print(f'\nSimulation version of {str(type(option))[17:-2]} with precision {pn} took {time_a - time_b} seconds to run!')

    


#Results:
'''
(Inheritance)
Vectorized version of EuropeanCallOption took 0.513 seconds to run!

Vectorized version of EuropeanPutOption took 0.496 seconds to run!

Vectorized version of AmericanCallOption took 0.503 seconds to run!

Vectorized version of AmericanPutOption took 0.497 seconds to run!
--------------------------------------------------------------------------------
(Inheritance)
Non-vectorized version of EuropeanCallOption took 1.477 seconds to run!

Non-vectorized version of EuropeanPutOption took 1.317 seconds to run!

Non-vectorized version of AmericanCallOption took 1.618 seconds to run!

Non-vectorized version of AmericanPutOption took 1.497 seconds to run!
--------------------------------------------------------------------------------
(No Inheritance)
Naive version of EuropeanCallOption took 2.141 seconds to run!

Naive version of EuropeanPutOption took 1.735 seconds to run!

Naive version of AmericanCallOption took 1.823 seconds to run!

Naive version of AmericanPutOption took 1.691 seconds to run!
'''


'''
Simulation version of EuropeanCallOption with precision 0.005 took 6.843 seconds to run!

Simulation version of EuropeanCallOption with precision 0.01 took 7.128 seconds to run!

Simulation version of EuropeanCallOption with precision 0.001 took 6.904 seconds to run!

Simulation version of EuropeanPutOption with precision 0.005 took 1.740 seconds to run!

Simulation version of EuropeanPutOption with precision 0.01 took 1.668 seconds to run!

Simulation version of EuropeanPutOption with precision 0.001 took 1.662 seconds to run!

'''