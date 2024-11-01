
from Potentials import LennardJones


k_b = 1#.38e-23 #J/K
T = 200 #Kelvin


# class LennardJones():

#     def __init__(self, eps0 = 5, sigma = 2**(-1/6)):
#         self.eps0 = eps0
#         self.sigma = sigma

#     def _V(self, distance):
#         V = 4*self.eps0*((self.sigma/distance)**12 - (self.sigma/distance)**6)
#         return V

#     def _dV_dr(self, r):
#         dV_dr = 4*self.eps0*(-12*(self.sigma/r)**13 + 6*(self.sigma/r)**7)
#         return dV_dr

#     def energy(self, positions): 
#         return np.sum(self._V(pdist(positions)))

#     def forces(self, pos):
#         diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
#         r = np.sqrt(np.sum(diff**2, axis=-1))
#         np.fill_diagonal(r, np.inf)
#         force_magnitude = self._dV_dr(r)
#         forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
#                         r[..., np.newaxis], axis=1)
#         return forces


class Potential():

    def __init__(self, V = 0, kT = 1, x_range = [-2,2], type = None, N_bins = 40, delta_x = 0.1, method = 'Random Walk'):
        self.V = 0
        self._kT = kT
        self.x_min, self.x_max = x_range
        self.type = type
        self._Q = None
        self._N_bins = N_bins
        self.x_is = []
        self.mcmc = None
        self.delta_x = delta_x
        self._method = method
        self.initial_point = 1
        self.current_mcmc_plot = {}
        
    

    def __str__(self):
        return f'Object with potential type: {self.type}'



    @property
    def kT(self):
        return self._kT

    @kT.setter
    def kT(self, value):
        self._kT = value
        self._Q = None 
        self._V_avg_direct = None
        self._C_V_fluctuations = None
        self._C_V_numerical = None
        self._V_avg_MC = None
        self.mcmc = None


    @property
    def Q(self):
        if self._Q is None:

            integrand = lambda x: np.exp(-self.V(x)/(self.kT))
            self._Q = quad(integrand, self.x_min, self.x_max)[0]
        return self._Q 
    
    
    
    @property
    def N_bins(self):
        return self._N_bins
    
    @N_bins.setter
    def N_bins(self, value):
        self._N_bins = value


    @property
    def mcmc_delta_x(self):
        return self.delta_x
    
    @mcmc_delta_x.setter
    def mcmc_delta_x(self, value):
        self.delta_x = value
        self.mcmc


    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value):
        self._method = value
        self.mcmc = None

    def P(self, x):
        P = np.exp(-self.V(x)/(self.kT))/self.Q
        return P
    
    

    #Plots of the potential and the probability distributions
    def plot_all(self,ax, N_mcmc = 1000):
        self.plot_V(ax)
        self.plot_P(ax)
        self.plot_mcmc(ax, N= N_mcmc)

        ax.legend()

    def plot_V(self, ax):

        x = np.linspace(self.x_min, self.x_max, 1000)
        ax.plot(x, self.V._V(x), label = f'{self.type} P(x)')

    def plot_V_mcmc(self, ax, N = 1000, initial_point = 0.1):
        if self.mcmc is None:
            self.sample_MCMC(N = N, initial_point=initial_point)
        ax.plot(self.mcmc, [self.V(point) for point in self.mcmc], 'o', alpha = 0.5, label = f'{self.type} P(x)')
    
    def plot_P(self, ax):
        x_for_hist = np.linspace(self.x_min, self.x_max, self._N_bins)
        ax.bar(x_for_hist, self.P(x_for_hist), width = (self.x_max - self.x_min)/self._N_bins -0.01, alpha = 0.8, label = f'P(x) {self.type}')
        ax.legend()
    
    def plot_mcmc(self, ax, N = 1000, initial_point = 0.1):
        if (self.mcmc is None) or (len(self.mcmc) < N):
            self.sample_MCMC(N = N, initial_point=initial_point)
    

        ax.hist(self.mcmc, bins = self._N_bins, alpha = 0.7, label = 'MCMC sampled values', density=True, range = (self.x_min, self.x_max))

    
    def plot_markov_chain(self, ax, N = 1000, initial_point = 0.1):
        if (self.mcmc is None) or (len(self.mcmc) < N):
            self.sample_MCMC(N = N, initial_point=initial_point)
        ax.plot(self.mcmc, range(len(self.mcmc)))

    def draw(self, ax, i, color = 'red'):
        if self.current_mcmc_plot.get(ax) is None:
            self.current_mcmc_plot[ax] = ax.plot(self.mcmc[i], self.V(self.mcmc[i]), 'o', color = color, alpha = 0.5)[0]
        self.current_mcmc_plot[ax].set_data(self.mcmc[i], self.V(self.mcmc[i]))
        return self.current_mcmc_plot[ax]
    


    #Monte Carlo sampling methods
    def _sample(self):
        x_i = np.random.uniform(self.x_min, self.x_max)
        Y_i = np.random.uniform(0,1)

        if Y_i < self.P(x_i):
            return x_i
        
        return None
    
    def sample_n_MC(self, N):
        x_is = []
        accepted = 0
        for _ in range(N):
            x_i = self._sample()
            if x_i is not None:
                x_is.append(x_i)
                accepted += 1
        print(f'Accepted {accepted} out of {N} samples: {accepted/N*100:.3g}% acceptance rate')
        self.x_is += x_is

    #Metropolis-Hastings MCMC sampling
    def sample_MCMC(self, N = 1000, initial_point = [0.1], break_point = None, write = True, calculator = LennardJones(), static_points = None, init_temp = 1e-4):
        shape = np.array(initial_point).shape
        temperature = init_temp
        match self.method:
            case 'Random Walk':
                random_walk = 1
                uniform = 0
            case 'Uniform':
                random_walk = 0
                uniform = 1
            case _:
                raise ValueError('Method must be "Random Walk" or "Uniform"')

        if self.mcmc is not None:
            if len(self.mcmc) > N:
                return
            else:
                initial_point = self.mcmc[-1]
                
            
        else:
            initial_point = initial_point
            self.mcmc = [initial_point]

        mcmc_chain = []
        accepted_points = 0
        current_point = initial_point

        for i in range(N - len(self.mcmc)):

            step_method_1 = self.mcmc_delta_x * np.random.randn(*shape) * random_walk
            #step_method_2 = (np.random.uniform(self.x_min, self.x_max) - current_point) * uniform 
            
            new_point = current_point + step_method_1# + step_method_2
            new_energy = calculator.energy(np.array([*new_point, *static_points]))
            current_energy = calculator.energy(np.array([*current_point, *static_points]))
            P_frac = ( np.exp(-new_energy/(self.kT)) ) / ( np.exp(-current_energy/(self.kT)) )#* np.exp(-(i*temperature/100))
            A = min(1, P_frac)
            if A > np.random.uniform(0,1):
                current_point = new_point
                accepted_points += 1
            mcmc_chain.append(current_point)
            if break_point is not None:
                if self.V(current_point) < break_point:
                    N = len(mcmc_chain)
                    break
            
        if write == True:
            print(f'Accepted {accepted_points} out of {N} samples: {(accepted_points)/(N)*100:.3g}% acceptance rate')
        self.mcmc = np.array(mcmc_chain)
        




    
    





        
        
        


    

