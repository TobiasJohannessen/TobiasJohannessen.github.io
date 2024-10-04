import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

k_b = 1#.38e-23 #J/K
T = 200 #Kelvin
        
    




        


   
class Potential():

    def __init__(self, kT = 1, x_range = [-2,2], type = None, N_bins = 40, delta_x = 0.1, method = 'Random Walk'):
        #self.V = V
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
    def plot_all(self,ax, N_mcmc = 1000, color = 'C0'):
        self.plot_V(ax, color = color)
        self.plot_P(ax, color = color)
        self.plot_mcmc(ax, N= N_mcmc, color = color)

        ax.legend()

    def plot_V(self, ax, color = 'C0'):

        x = np.linspace(self.x_min, self.x_max, 1000)
        ax.plot(x, self.V(x), label = f'{self.type} P(x)', color = color)
    
    def plot_P(self, ax, color = 'C0'):
        x_for_hist = np.linspace(self.x_min, self.x_max, self._N_bins)
        ax.bar(x_for_hist, self.P(x_for_hist), width = (self.x_max - self.x_min)/self._N_bins -0.01, alpha = 0.8, label = f'P(x) {self.type}', color = color)
        ax.legend()
    
    def plot_mcmc(self, ax, N = 1000, initial_point = 0.1, color = 'C0'):
        if (self.mcmc is None) or (len(self.mcmc) < N):
            self.sample_MCMC(N = N, initial_point=initial_point)
    

        ax.hist(self.mcmc, bins = self._N_bins, alpha = 0.7, label = f'MCMC sampled values ({self.method})', density=True, range = (self.x_min, self.x_max), color = color)
    
    
    



   
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

        #Method 3 to find: <V>
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
    def sample_MCMC(self, N = 1000, initial_point = [0.1], break_point = None, write = True, static_points = None):
        shape = np.array(initial_point).shape
        
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
            new_energy = self.V(np.array([*new_point, *static_points]))
            current_energy = self.V(np.array([*current_point, *static_points]))
            P_frac = ( np.exp(-new_energy/(self.kT)) ) / ( np.exp(-current_energy/(self.kT)) )
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
        

    def _velocity_Verlet(self, r, v, dt = 0.1, static_points = []):

        d = len(r) #Dimension of the moving system
        try:
            r_new = r + dt * v + dt**2 * 1/2 * self.force(np.array([*r, *static_points]))[:d:]
        except ValueError:
            r_new = r + dt * v + dt**2 * 1/2 * self.force(np.array([r, *static_points]))[:d:]
        v_new = v + 1/2 * (self.force(np.array([*r, *static_points]))[:2:] + self.force(np.array([*r_new, *static_points]))[:d:] )* dt
        return r_new, v_new

    def N_velocity_Verlet(self, r0, v0, N, dt = 0.01, static_points = []):
        r = r0
        v = v0
        rs = []
        vs = []
        for _ in range(N):
            r, v = self._velocity_Verlet(r = r, v = v, dt = dt, static_points = static_points)
            rs.append(r)
            vs.append(v)
        return rs, vs


    def constant_temp_MD(self, r0,N = 10000, m = 1, dt = 0.01, static_points = []):
        r = r0
        shape = np.array(r0).shape
        rs = []
        for _ in range(N):
            v = np.random.normal(0,np.sqrt(self.kT/m), shape)
            rs_new, vs_new = self.N_velocity_Verlet(r0 = r, v0 = v, N = 50, dt = dt, static_points = static_points)
            r = rs_new[-1]
            rs += [r]
        
        V_rs = [self.V(r) for r in rs]

        self.V_avg_MD = np.mean(V_rs)

        self.C_V_MD = np.var(V_rs)/(self.kT**2)
        
        return rs,  self.V_avg_MD, self.C_V_MD

    def line_search(self, r0, N = 1000, dt = 0.01, static_points = [], tol = 1e-3):
        r = r0
        
        rs = [r0]
        for _ in range(N):
            try:
                d = len(r)
                F= self.force(np.array([*r, *static_points]))[:d:]
            except ValueError:
                d = 1
                F= self.force(np.array([r, *static_points]))[:d:]
            F_norm = np.linalg.norm(F)
            F_direction = F/F_norm
            a0 = 1e-3
            energy = lambda alpha: self.V(r + alpha*F_direction)
            alpha = minimize(energy, a0).x
            r_new = r + alpha*F_direction
            if energy(r_new)/energy(r) < tol:
                break
            r = r_new
            rs.append(r)
        return np.array(rs)
        
        
        


    

    def thermodynamics(self):
        self._V_avg_direct = self.direct_integration()
        self._C_V_fluctuations = self.fluctuations()
        self._C_V_numerical = self.numerical_derivative()
        self._V_avg_MC = self.direct_MC()
        
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

    def __init__(self, k = 1, x0 = 0, x_range = [-2, 2], kT = 1, N_bins = 40):
        self.k = k
        self.x0 = x0
        self.type = 'Harmonic'
        
        super().__init__(type = self.type, x_range = x_range, kT = kT, N_bins=N_bins)

    def _force(self, x):
        return -self.k*(x-self.x0)

    def V(self, x):
        return 0.5*self.k*(x-self.x0)**2

class CustomPotential(Potential):

    def __init__(self, k = 1, x0 = 0, x1 = 1, A = 1, B = 1, x_range=[-2,2], kT = 0.15, N_bins = 40):

        self.k = k
        self.x0 = x0
        self.x1 = x1
        self.type = 'Harmonic + Exponential'
        self.A = A
        self.B = B
        super().__init__( type = self.type, x_range = x_range, kT = kT, N_bins=N_bins)

    def force(self, x):
        return -self.k*(x-self.x0) + 2*self.A/self.B**2 * (x-self.x1) * np.exp(-((x-self.x1)/self.B)**2)

    
    def V(self, x):
        return 0.5*self.k*(x-self.x0)**2 + self.A * np.exp(- ((x-self.x1)/self.B)**2)


    def E_pot_numeric(self, x):
        E_pot_0 = self.V(self.x0)
        integrand = lambda x_: self.force(x_)
        E_pot_1 = quad(integrand, self.x0, x)[0]
        return E_pot_0 - E_pot_1


class LennardJones(Potential):

    def __init__(self, eps0 = 5, sigma = 2**(-1/6), x_range = [0.8, 2], kT = 0.15, N_bins = 40):
        self.eps0 = eps0
        self.sigma = sigma
        self.type = 'Lennard-Jones'
        super().__init__( type = self.type, x_range = x_range, kT = kT, N_bins=N_bins)

    def _V(self, distance):
        V = 4*self.eps0*((self.sigma/distance)**12 - (self.sigma/distance)**6)
        return V

    def _dV_dr(self, r):
        dV_dr = 4*self.eps0*(-12*(self.sigma/r)**13 + 6*(self.sigma/r)**7)
        return dV_dr

    def V(self, positions): 
        return np.sum(self._V(pdist(positions)))

    def force(self, pos):
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(r, np.inf)
        force_magnitude = self._dV_dr(r)
        forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                        r[..., np.newaxis], axis=1)
        return forces



