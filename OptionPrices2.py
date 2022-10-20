# Author(s): Maysam Khodayari

import time
import numpy as np
from matplotlib import pyplot as plt
from numpy import log,exp,sqrt
from scipy.stats import norm   # for Normal CDF

class EuropeanCallOption:
    def __init__(self, S0, K, r, sigma, T):
        self._S0 = S0        # initial stock price
        self._K = K          # strike price
        self._r = r          # risk-free rate
        self._sigma = sigma  # volatility
        self._T = T          # expiration time


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
        
    
    # Closed-form formulas: Black-Scholes-Merton
        def BSMPrice(self):        
            d1 = (log(self._S0 / self._K) + (self._r + (self._sigma**2) / 2 ) * (self._T)) / (self._sigma * self._T ** 0.5)
            d2 = d1 - self._sigma * self._T ** 0.5        
            C0 = (norm.cdf(d1) *  self._S0) - (norm.cdf(d2) * self._K * exp(-self._r * self._T))
            return C0
    
    # Binomial Tree
    def binomialPrice(self, num_intervals):
        # local variables used by this function
        deltaT = self._T / num_intervals
        u = exp(self._sigma * deltaT ** 0.5)
        d = 1 / u
        a = exp(self._r * deltaT)
        p = (a - d) / (u - d)
        q = 1 - p
        # abbreviated name
        ni = num_intervals
        # fill tree with all 0.0s
        binom_tree = []
        for i in range(ni + 1):
            ith_row = [self._PriceNode()
                              for j in range(i+1)]
            binom_tree.append(ith_row)
        if ni < 10:
            print('\nAfter filled in with all 0.0:')
            self.binomialTreePretty(binom_tree)
        # fill tree with stock prices
        for i in range(ni + 1):
            for j in range(i + 1):
                binom_tree[i][j]._stock_price = (
                        self._S0 * u ** j * d ** (i-j))
        if ni < 10:
            print('\nAfter filled in with stock prices:')
            self.binomialTreePretty(binom_tree)
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(binom_tree[ni][j]._stock_price - self._K,
                            0.0))
        if ni < 10:
            print('\nAfter filled in with terminal option values:')
            self.binomialTreePretty(binom_tree)
        # Now work backwards, filling in the
        # option prices in the rest of the tree
        # Your code here
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = ((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a

        if ni < 10:
            print('\nAfter filled in with all option values:')
            self.binomialTreePretty(binom_tree)
        return binom_tree[0][0]._option_price

    def binomialTreePretty(self, binom_tree):
        nlevels = len(binom_tree)
        if nlevels < 10:
            print('\nBinomialTree with', nlevels - 1, 'time steps:')
            for row in range(nlevels):
                print('\nStock:  ', end='')
                ncols = len(binom_tree[row])
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._stock_price), end='')
                print('')
                print('Option: ', end='')
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._option_price), end='')
                print('')

if __name__ == '__main__':
    
    ec = EuropeanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)

    print(ec)


    print('Binomial Tree Euro call price, 5 time intervals: '
          + '${:.4f}'.format(ec.binomialPrice(5)))



    for steps in [10,20,50,100,200,500,1000]:
        print(('Binomial Tree Euro call price, {:d} time intervals: '
              + '${:.4f}').format(steps, ec.binomialPrice(steps)))




    for S0 in [5, 50, 500]:
        # strike price equals stock price
        ecS0 = EuropeanCallOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, ecS0)
        print(('Binomial Tree Euro call price, 1000 time intervals: '
              + '${:.4f}').format(ecS0.binomialPrice(1000)))



    print(ec)   # same as above
    print(('BSM Euro call price: '
              + '${:.4f}').format(ec.BSMPrice()))
              
    for S0 in [5, 50, 500]:
        # strike price equals stock price
        ecS0 = EuropeanCallOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, ecS0)
        print(('BSM Euro call price: '
              + '${:.4f}').format(ecS0.BSMPrice()))



vl = [.05, .10, .20, .40, .80, 1.60, 3.20]
res = []
for v in vl:
    a = EuropeanCallOption(50.0, 50.0, 0.1, v, 0.4167)
    res.append(a.binomialPrice(1000))

plt.title('Plain Vanilla European Call Prices vs. Volatility')
plt.xlabel("volatility")
plt.ylabel("call option price")
plt.plot(vl, res)
plt.show()    



class EuropeanPutOption:
    def __init__(self, S0, K, r, sigma, T):
        self._S0 = S0        # initial stock price
        self._K = K          # strike price
        self._r = r          # risk-free rate
        self._sigma = sigma  # volatility
        self._T = T          # expiration time

# Inner class used by the binomial tree model
    class _PriceNode:
        def __init__(self):
            self._stock_price = 0
            self._option_price = 0  
    

# EuropeanPutOption methods:
    def __str__(self):
        return ('EuropeanPutOption:\n'
                + '  S0: ' + str(self._S0)
                + '  K: ' + str(self._K)
                + '  r: ' + str(self._r)
                + '  sigma: ' + str(self._sigma)
                + '  T: ' + str(self._T))
    
    
    
    # Closed-form formulas: Black-Scholes-Merton
    def BSMPrice(self):        
        d1 = (log(self._S0 / self._K) + (self._r + (self._sigma**2) / 2 ) * (self._T)) / (self._sigma * self._T ** 0.5)
        d2 = d1 - self._sigma * self._T ** 0.5        
        C0 = (norm.cdf(-d2) * self._K * exp(-self._r * self._T)) - (norm.cdf(-d1) *  self._S0)
        return C0
    
    # Binomial Tree    
    def binomialPrice(self, num_intervals):
        # local variables used by this function
        deltaT = self._T / num_intervals
        u = exp(self._sigma * deltaT ** 0.5)
        d = 1 / u
        a = exp(self._r * deltaT)
        p = (a - d) / (u - d)
        q = 1 - p
        # abbreviated name
        ni = num_intervals
        # fill tree with all 0.0s
        binom_tree = []
        for i in range(ni + 1):
            ith_row = [self._PriceNode()
                              for j in range(i+1)]
            binom_tree.append(ith_row)
        if ni < 10:
            print('\nAfter filled in with all 0.0:')
            self.binomialTreePretty(binom_tree)
        # fill tree with stock prices
        for i in range(ni + 1):
            for j in range(i + 1):
                binom_tree[i][j]._stock_price = (
                        self._S0 * u ** j * d ** (i-j))
        if ni < 10:
            print('\nAfter filled in with stock prices:')
            self.binomialTreePretty(binom_tree)
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(self._K - binom_tree[ni][j]._stock_price,
                            0.0))
        if ni < 10:
            print('\nAfter filled in with terminal option values:')
            self.binomialTreePretty(binom_tree)
        # Now work backwards, filling in the
        # option prices in the rest of the tree
        # Your code here
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = ((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a

        if ni < 10:
            print('\nAfter filled in with all option values:')
            self.binomialTreePretty(binom_tree)
        return binom_tree[0][0]._option_price

    def binomialTreePretty(self, binom_tree):
        nlevels = len(binom_tree)
        if nlevels < 10:
            print('\nBinomialTree with', nlevels - 1, 'time steps:')
            for row in range(nlevels):
                print('\nStock:  ', end='')
                ncols = len(binom_tree[row])
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._stock_price), end='')
                print('')
                print('Option: ', end='')
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._option_price), end='')
                print('')
            print('')

if __name__ == '__main__':
    
    ep = EuropeanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)

    print(ec)

    print('Binomial Tree Euro put price, 5 time intervals: '
          + '${:.4f}'.format(ep.binomialPrice(5)))


    for steps in [10,20,50,100,200,500,1000]:
        print(('Binomial Tree Euro put price, {:d} time intervals: '
              + '${:.4f}').format(steps, ep.binomialPrice(steps)))


    for S0 in [5, 50, 500]:
        # strike price equals stock price
        epS0 = EuropeanPutOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, epS0)
        print(('Binomial Tree Euro put price, 1000 time intervals: '
              + '${:.4f}').format(epS0.binomialPrice(1000)))

    print(ep)   # same as above
    print(('BSM Euro put price: '
              + '${:.4f}').format(ep.BSMPrice()))
              
    for S0 in [5, 50, 500]:
        # strike price equals stock price
        epS0 = EuropeanPutOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, epS0)
        print(('BSM Euro put price: '
              + '${:.4f}').format(epS0.BSMPrice()))

vl = [.05, .10, .20, .40, .80, 1.60, 3.20]
res = []
for v in vl:
    a = EuropeanPutOption(50.0, 50.0, 0.1, v, 0.4167)
    res.append(a.binomialPrice(1000))

plt.title('Plain Vanilla European Put Prices vs. Volatility')
plt.xlabel("volatility")
plt.ylabel("put option price")
plt.plot(vl, res)
plt.show()


''' The ERI output exactly match BSMPrice() '''

# 2.a
class AmericanCallOption:
    def __init__(self, S0, K, r, sigma, T):
        self._S0 = S0        # initial stock price
        self._K = K          # strike price
        self._r = r          # risk-free rate
        self._sigma = sigma  # volatility
        self._T = T          # expiration time


    # Inner class used by the binomial tree model
    class _PriceNode:
        def __init__(self):
            self._stock_price = 0
            self._option_price = 0       

# AmericanCallOption methods:
    def __str__(self):
        return ('AmericanCallOption:\n'
                + '  S0: ' + str(self._S0)
                + '  K: ' + str(self._K)
                + '  r: ' + str(self._r)
                + '  sigma: ' + str(self._sigma)
                + '  T: ' + str(self._T))
        
    
  
    # Binomial Tree
    def binomialPrice(self, num_intervals):
        # local variables used by this function
        deltaT = self._T / num_intervals
        u = exp(self._sigma * deltaT ** 0.5)
        d = 1 / u
        a = exp(self._r * deltaT)
        p = (a - d) / (u - d)
        q = 1 - p
        # abbreviated name
        ni = num_intervals
        # fill tree with all 0.0s
        binom_tree = []
        for i in range(ni + 1):
            ith_row = [self._PriceNode()
                              for j in range(i+1)]
            binom_tree.append(ith_row)
        if ni < 10:
            print('\nAfter filled in with all 0.0:')
            self.binomialTreapretty(binom_tree)
        # fill tree with stock prices
        for i in range(ni + 1):
            for j in range(i + 1):
                binom_tree[i][j]._stock_price = (
                        self._S0 * u ** j * d ** (i-j))
        if ni < 10:
            print('\nAfter filled in with stock prices:')
            self.binomialTreapretty(binom_tree)
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(binom_tree[ni][j]._stock_price - self._K,
                            0.0))
        if ni < 10:
            print('\nAfter filled in with terminal option values:')
            self.binomialTreapretty(binom_tree)
        # Now work backwards, filling in the
        # option prices in the rest of the tree
        # Your code here
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = max(((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a,\
                                                     binom_tree[i][j]._stock_price - self._K )

        if ni < 10:
            print('\nAfter filled in with all option values:')
            self.binomialTreapretty(binom_tree)
        return binom_tree[0][0]._option_price

    def binomialTreapretty(self, binom_tree):
        nlevels = len(binom_tree)
        if nlevels < 10:
            print('\nBinomialTree with', nlevels - 1, 'time staps:')
            for row in range(nlevels):
                print('\nStock:  ', end='')
                ncols = len(binom_tree[row])
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._stock_price), end='')
                print('')
                print('Option: ', end='')
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._option_price), end='')
                print('')
            print('')

if __name__ == '__main__':
    
    ac = AmericanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)

    print(ac)


    print('Binomial Tree American call price, 5 time intervals: '
          + '${:.4f}'.format(ac.binomialPrice(5)))



    for staps in [10,20,50,100,200,500,1000]:
        print(('Binomial Tree American call price, {:d} time intervals: '
              + '${:.4f}').format(staps, ac.binomialPrice(staps)))




    for S0 in [5, 50, 500]:
        # strike price equals stock price
        acS0 = AmericanCallOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, acS0)
        print(('Binomial Tree American call price, 1000 time intervals: '
              + '${:.4f}').format(acS0.binomialPrice(1000)))



    print(ac)   # same as above
    print(('BSM American call price: '
              + '${:.4f}').format(ac.BSMPrice()))
              
    for S0 in [5, 50, 500]:
        # strike price equals stock price
        acS0 = AmericanCallOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, acS0)
        print(('BSM American call price: '
              + '${:.4f}').format(acS0.BSMPrice()))



vl = [.05, .10, .20, .40, .80, 1.60, 3.20]
res = []
for v in vl:
    a = AmericanCallOption(50.0, 50.0, 0.1, v, 0.4167)
    res.append(a.binomialPrice(1000))

plt.title('Plain Vanilla American Call Prices vs. Volatility')
plt.xlabel("volatility")
plt.ylabel("call option price")
plt.plot(vl, res)
plt.show()    


class AmericanPutOption:
    def __init__(self, S0, K, r, sigma, T):
        self._S0 = S0        # initial stock price
        self._K = K          # strike price
        self._r = r          # risk-free rate
        self._sigma = sigma  # volatility
        self._T = T          # expiration time

# Inner class used by the binomial tree model
    class _PriceNode:
        def __init__(self):
            self._stock_price = 0
            self._option_price = 0  
    

# AmericanPutOption methods:
    def __str__(self):
        return ('AmericanPutOption:\n'
                + '  S0: ' + str(self._S0)
                + '  K: ' + str(self._K)
                + '  r: ' + str(self._r)
                + '  sigma: ' + str(self._sigma)
                + '  T: ' + str(self._T))
    
    
    
    # Binomial Tree    
    def binomialPrice(self, num_intervals):
        # local variables used by this function
        deltaT = self._T / num_intervals
        u = exp(self._sigma * deltaT ** 0.5)
        d = 1 / u
        a = exp(self._r * deltaT)
        p = (a - d) / (u - d)
        q = 1 - p
        # abbreviated name
        ni = num_intervals
        # fill tree with all 0.0s
        binom_tree = []
        for i in range(ni + 1):
            ith_row = [self._PriceNode()
                              for j in range(i+1)]
            binom_tree.append(ith_row)
        if ni < 10:
            print('\nAfter filled in with all 0.0:')
            self.binomialTreapretty(binom_tree)
        # fill tree with stock prices
        for i in range(ni + 1):
            for j in range(i + 1):
                binom_tree[i][j]._stock_price = (
                        self._S0 * u ** j * d ** (i-j))
        if ni < 10:
            print('\nAfter filled in with stock prices:')
            self.binomialTreapretty(binom_tree)
        # fill in terminal node option prices
        for j in range(ni + 1):
            binom_tree[ni][j]._option_price = (
                max(self._K - binom_tree[ni][j]._stock_price,
                            0.0))
        if ni < 10:
            print('\nAfter filled in with terminal option values:')
            self.binomialTreapretty(binom_tree)
        # Now work backwards, filling in the
        # option prices in the rest of the tree
        # Your code here
        for i in range(ni - 1, -1, -1):
            for j in range(0, i+1):
                binom_tree[i][j]._option_price = max(((p * binom_tree[i + 1][j + 1]._option_price)\
                                                  + (q * binom_tree[i + 1][j]._option_price)) / a,\
                                                     self._K - binom_tree[i][j]._stock_price)

        if ni < 10:
            print('\nAfter filled in with all option values:')
            self.binomialTreapretty(binom_tree)
        return binom_tree[0][0]._option_price

    def binomialTreapretty(self, binom_tree):
        nlevels = len(binom_tree)
        if nlevels < 10:
            print('\nBinomialTree with', nlevels - 1, 'time staps:')
            for row in range(nlevels):
                print('\nStock:  ', end='')
                ncols = len(binom_tree[row])
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._stock_price), end='')
                print('')
                print('Option: ', end='')
                for col in range(ncols):
                    print('{:8.4f}'.format(binom_tree[row][col]._option_price), end='')
                print('')
            print('')

if __name__ == '__main__':
    
    ap = AmericanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)

    print(ac)

    print('Binomial Tree American put price, 5 time intervals: '
          + '${:.4f}'.format(ap.binomialPrice(5)))


    for staps in [10,20,50,100,200,500,1000]:
        print(('Binomial Tree American put price, {:d} time intervals: '
              + '${:.4f}').format(staps, ap.binomialPrice(staps)))


    for S0 in [5, 50, 500]:
        # strike price equals stock price
        apS0 = AmericanPutOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, apS0)
        print(('Binomial Tree American put price, 1000 time intervals: '
              + '${:.4f}').format(apS0.binomialPrice(1000)))

    print(ap)   # same as above
    print(('BSM American put price: '
              + '${:.4f}').format(ap.BSMPrice()))
              
    for S0 in [5, 50, 500]:
        # strike price equals stock price
        apS0 = AmericanPutOption(S0, S0, 0.1, 0.4, 0.4167)
        print('With S0, K ==', S0, apS0)
        print(('BSM American put price: '
              + '${:.4f}').format(apS0.BSMPrice()))

vl = [.05, .10, .20, .40, .80, 1.60, 3.20]
res = []
for v in vl:
    a = AmericanPutOption(50.0, 50.0, 0.1, v, 0.4167)
    res.append(a.binomialPrice(1000))

plt.title('Plain Vanilla American Put Prices vs. Volatility')
plt.xlabel("volatility")
plt.ylabel("put option price")
plt.plot(vl, res)
plt.show()



'''I observed that for the data set: K, S0 = 50,
 r = 0.1, sigma = 0.4, T = 0.4167), American and European has 
the same value ($6.12 but American Put ($4.28) 
is more expensive than European ($4.07)
I compared my results with https://financial-calculators.com/options-calculator
and its results exactly match with mine!'''

ec = EuropeanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ep = EuropeanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ac = AmericanCallOption(50.0, 50.0, 0.1, 0.4, 0.4167)
ap = AmericanPutOption(50.0, 50.0, 0.1, 0.4, 0.4167)

options = [ec, ep, ac, ap]

for option in options:
    time_b = time.time()
    option.binomialPrice(1000)
    time_a = time.time()
    print(f'\nNaive version of {str(type(option))[17:-2]} took {time_a - time_b} seconds to run!')

