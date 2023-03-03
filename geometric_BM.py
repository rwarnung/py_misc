import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(s_0, mu, sigma, T, N, n_sims = 10**3, random_seed = 1000):
    """ simulated stock prices using  gbm """

    np.random.seed(random_seed)
    dt = T/N

    # define Brownian increments
    dW = np.random.normal(scale=np.sqrt(dt), size=(n_sims, N+1))

    # sim process; cumsum over rows
    S_t = s_0 * np.exp(np.cumsum( (mu-sigma**2/2)*dt + sigma*dW, axis=1 ) )
    S_t[:, 0] = s_0

    return(S_t)

## simulate 10 paths

N = 100
n_sims = 10**3
T = 2
mu = 0.02
sigma = 0.1
BARRIER = 120
K = 110
r = mu

my_stock = simulate_gbm(s_0 = 100, mu = mu, sigma = sigma, T = T, N = N, n_sims = n_sims)

## small example

n_sims = 4
my_stock = simulate_gbm(s_0 = 100, mu = mu, sigma = sigma, T = T, N = N, n_sims = n_sims)
## same
plt.axhline(y=BARRIER, color = 'r', linestyle = '-',  linewidth=1.5, label = "Barrier")
plt.axhline(y=K, color = 'black', linestyle = '-',  linewidth=1.5, label = "strike")
plt.xlim(0, N) ## N steps
plt.plot(my_stock.T, linewidth=0.25, label="stock price") ## plot transposed
plt.legend()
plt.show()

## normal size

n_sims = 10**3
my_stock = simulate_gbm(s_0 = 100, mu = mu, sigma = sigma, T = T, N = N, n_sims = n_sims)

## plot the paths

plt.axhline(y=BARRIER, color = 'r', linestyle = '-',  linewidth=1.5)
plt.axhline(y=K, color = 'black', linestyle = '-',  linewidth=1.5)
plt.xlim(0, N) ## N steps
plt.plot(my_stock.T, linewidth=0.25) ## plot transposed
plt.legend(['BARRIER','strike', 'Paths of the stock price'])
plt.show()



## define pay-off; call if the BARRIER was reached
max_value_per_path = np.max(my_stock, axis=1)

payoff = np.where(
    max_value_per_path > BARRIER, ## if once the barrier was breached
    np.maximum(0, my_stock[:,-1] - K), ## call pay-off at maturity
    0 ## else zero
)
payoff.shape

## discounting
disc_payoff = payoff*np.exp(-r*T)
premium =  np.mean(disc_payoff)