import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import quad


k_b = 1#.38e-23 #J/K
T = 200 #Kelvin


class Potential():

    def __init__(self, V, kT = 1, x_range = [-2,2], type = None, N_bins = 40, delta_x = 0.1, method = 'Random Walk'):
        self.V = V
        self._kT = kT
        self.x_min, self.x_max = x_range
        self.type = type
        self._Q = None
        self._N_bins = N_bins
        self.x_is = []
        self.V_avg_direct = None
        self.C_V_fluctuations = None
        self.C_V_numerical = None
        self.V_avg_MC = None
        self.mcmc = None
        self.delta_x = delta_x
        self._method = method
        
    

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
    def V_avg_direct(self):
        if self._V_avg_direct is None:
            self._V_avg_direct = self.direct_integration()
        return self._V_avg_direct
    
    @V_avg_direct.setter
    def V_avg_direct(self, value):
        self._V_avg_direct = value
    

    @property
    def C_V_fluctuations(self):
        if self._C_V_fluctuations is None:
            self._C_V_fluctuations = self.fluctuations()
        return self._C_V_fluctuations
    
    @C_V_fluctuations.setter
    def C_V_fluctuations(self, value):
        self._C_V_fluctuations = value
    

    @property
    def C_V_numerical(self):
        if self._C_V_numerical is None:
            self._C_V_numerical = self.numerical_derivative()
        return self._C_V_numerical
    
    @C_V_numerical.setter
    def C_V_numerical(self, value):
        self._C_V_numerical = value
    

    @property
    def V_avg_MC(self):
        if self._V_avg_MC is None:
            self._V_avg_MC = self.direct_MC()
        return self._V_avg_MC

    @V_avg_MC.setter
    def V_avg_MC(self, value):
        self._V_avg_MC = value

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
        ax.plot(x, self.V(x), label = f'{self.type} P(x)')
    
    def plot_P(self, ax):
        x_for_hist = np.linspace(self.x_min, self.x_max, self._N_bins)
        ax.bar(x_for_hist, self.P(x_for_hist), width = (self.x_max - self.x_min)/self._N_bins -0.01, alpha = 0.8, label = f'P(x) {self.type}')
        ax.legend()
    
    def plot_mcmc(self, ax, N = 1000, initial_point = 0.1):
        if (self.mcmc is None) or (len(self.mcmc) < N):
            self.sample_MCMC(N = N, initial_point=initial_point)
    

        ax.hist(self.mcmc, bins = self._N_bins, alpha = 0.7, label = 'MCMC sampled values', density=True, range = (self.x_min, self.x_max))
    



   
    #Methods to calculate the expectation value and  of the potential
    def direct_integration(self, V = None, type = None):
        if type is None:
            type = self.type

        match type:
            case 'Harmonic':
                return self.direct_integration_harmonic()
            case _:
                return self.direct_integration_general(V = V)


    def direct_integration_harmonic(self):
        exp_val = 1/2 * self.kT
        

        return exp_val
    
    def direct_integration_general(self, V = None):

        if V is None:
            V = self.V

        integrand = lambda x: V(x)*np.exp(-V(x)/(self.kT))
        exp_val = 1/self.Q * quad(integrand, self.x_min, self.x_max)[0]

        return exp_val
    
    
    

    def numerical_derivative(self, dT = 0.001):

        #Method 1 to find: C_V = d<E>/dT 
        #Use definition of derivative to calculate the derivative of the expectation value


        #Find the expectation value at two different temperatures a small distance apart
        exp_val1 = self.direct_integration()

        self.kT += dT

        exp_val2 = self.direct_integration()

        self.kT -= dT

        C_V = (exp_val2 - exp_val1)/dT
        return C_V

    def fluctuations(self):
        #Method 2 to find: C_V = d<E>/dT 
        #Use C_V = -1/kT^2 * (<E^2> - <E>^2)
        
        integrand1 = lambda x: self.V(x)*np.exp(-self.V(x)/(self.kT))
        exp_val1 = 1/self.Q * quad(integrand1, self.x_min, self.x_max)[0]

        integrand2 = lambda x: self.V(x)**2*np.exp(-self.V(x)/(self.kT))
        exp_val2 = 1/self.Q * quad(integrand2, self.x_min, self.x_max)[0]

        C_V = 1/self.kT**2 * (exp_val2 - exp_val1**2)

        return C_V
    
    def direct_MC(self, N = 100000):

        #Method 3 to find: C_V = d<E>/dT 
        #Use Monte Carlo sampling to calculate the expectation value

        self.sample_n_MC(N)
        exp_val = 1/len(self.x_is) * np.sum([self.V(x_i) for x_i in self.x_is])

        return exp_val
    

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
    def sample_MCMC(self, N = 1000, initial_point = 0.1):

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
        current_point = initial_point
    
        for _ in range(N - len(self.mcmc)):

            step_method_1 = self.mcmc_delta_x * np.random.normal(0,1) * random_walk
            step_method_2 = (np.random.uniform(self.x_min, self.x_max) - current_point) * uniform 
            
            new_point = current_point + step_method_1 + step_method_2
           
            A = min(1, self.P(new_point)/self.P(current_point))
            if A > np.random.uniform(0,1):
                current_point = new_point
                mcmc_chain.append(current_point)
            

        print(f'Accepted {len(mcmc_chain) - 1} out of {N - len(self.mcmc)} samples: {(len(mcmc_chain) - 1)/(N - len(self.mcmc))*100:.3g}% acceptance rate')
        self.mcmc = mcmc_chain
        
    
    

    def thermodynamics(self):
        self.V_avg_direct = self.direct_integration()
        self.C_V_fluctuations = self.fluctuations()
        self.C_V_numerical = self.numerical_derivative()
        self.V_avg_MC = self.direct_MC()
        
        print(f'Heat capacity for {self.type} potential:')
        print(f'Direct integration: {self.V_avg_direct:.3g}')
        print(f'Fluctuations: {self.C_V_fluctuations:.3g}')
        print(f'Numerical derivative: {self.C_V_numerical:.3g}')
        print(f'Monte Carlo: {self.V_avg_MC:.3g}')

        return self._V_avg_direct, self._C_V_fluctuations, self._C_V_numerical, self._V_avg_MC
    

    def non_boltzmann_sample(self, other, N = 100000):
        if len(self.x_is) <= N:
            self.sample_n_MC(N)
        
        
        

        Q_integrand = lambda x: np.exp(-other.V(x) /(self.kT))
        Q_sampled = 1/len(self.x_is) * np.sum([Q_integrand(x_i) for x_i in self.x_is])
        
        obs_integrand = lambda x: other.V(x)*np.exp(-other.V(x)/(self.kT))
        obs_sampled = 1/len(self.x_is) * np.sum([obs_integrand(x_i) for x_i in self.x_is])

        return obs_sampled/Q_sampled


    


        
    
class HarmonicOscillator(Potential):

    def __init__(self, k = 1, x_0 = 0, x_range = [-2, 2], kT = 1, N_bins = 40):
        self.k = k
        self.x_0 = x_0
        self.type = 'Harmonic'
        self.V = lambda x: 0.5*k*(x-x_0)**2
        super().__init__(self.V, type = self.type, x_range = x_range, kT = kT, N_bins=N_bins)




        
        
        


    


