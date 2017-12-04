from ..frame import *
from .simulation_class import simulation_class


class stochastic_volatility(simulation_class):
    ''' Class to generate simulated paths based on
    the Heston (1993) stochastic volatility model.
    Attributes
    ==========
    name : string
        name of the object
    mar_env : instance of market_environment
        market environment data for simulation
    corr : boolean
        True if correlated with other model object
    Methods
    =======
    update :
        updates parameters
    generate_paths :
        returns Monte Carlo paths given the market environment
    get_volatility_values :
        returns array with simulated volatility paths
    '''

    def __init__(self, name, mar_env, corr=False):
        super(stochastic_volatility, self).__init__(name, mar_env, corr)
        try:
            self.kappa = mar_env.get_constant('kappa')
            self.theta = mar_env.get_constant('theta')
            self.vol_vol = mar_env.get_constant('vol_vol')

            self.rho = mar_env.get_constant('rho')
            self.leverage = np.linalg.cholesky(
                np.array([[1.0, self.rho], [self.rho, 1.0]]))

            self.volatility_values = None
        except:
            print('Error parsing market environment.')

    def update(self, pricing_date=None, initial_value=None, volatility=None, 
    vol_vol=None, kappa=None, theta=None, final_date=None):
        if pricing_date is not None:
            self.pricing_date = pricing_date
            self.time_grid = None
            self.generate_time_grid()
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if vol_vol is not None:
            self.vol_vol = vol_vol
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
            self.time_grid = None
        self.instrument_values = None
        self.volatility_values = None


    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        va = np.zeros_like(paths)
        va_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        va[0] = self.volatility ** 2
        va_[0] = self.volatility ** 2
        if self.correlated is False:
            sn1 = sn_random_numbers((1, M, I),
                                    fixed_seed=fixed_seed)
        else:
            # Pseudo random numbers for the monte carlo
            sn1 = self.random_numbers
        
        # Pseudo-random numbers for the stochastic volatility
        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)

        
        forward_rates = self.discount_curve.get_forward_rates(
            self.time_grid, self.paths, dtobjects=True)[1]

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            
            # rat = the pair of random numbers we need for each path
            rat = np.array([ran, sn2[t]])
            rat = np.dot(self.leverage, rat)

            # Heston Model of Stochastic volatility
            # instantaneous vol will move from intial vol (va_[0]**2) to long term variance (theta)
            # at the rate of kappa as the path gets further
            va_[t] = (va_[t - 1] + self.kappa * (self.theta - np.maximum(0, va_[t - 1])) * dt \
                     + np.sqrt(np.maximum(0, va_[t - 1])) * self.vol_vol * np.sqrt(dt) * rat[1])
            va[t] = np.maximum(0, va_[t])

            # Assume an average of the staring and ending forward rates over the period
            rt = (forward_rates[t - 1] + forward_rates[t]) / 2
            
            # Normal Geometric Brownian motion after stochastic volatility
            paths[t] = paths[t - 1] * (np.exp((rt - 0.5 * va[t]) * dt +
                       np.sqrt(va[t]) * np.sqrt(dt) * rat[0]))
            
            # moment matching stoch vol part
            # moment matching ensures std dev = 1 for normal distribution
            paths[t] -= np.mean(paths[t - 1] * np.sqrt(va[t]) * math.sqrt(dt) * rat[0])

        self.instrument_values = paths
        self.volatility_values = np.sqrt(va)

    def get_volatility_values(self):
        if self.volatility_values is None:
            self.generate_paths(self)
        return self.volatility_values