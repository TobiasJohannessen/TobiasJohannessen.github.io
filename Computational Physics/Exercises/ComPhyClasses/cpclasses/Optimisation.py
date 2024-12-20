
from Potentials import LennardJones


k_b = 1#.38e-23 #J/K
T = 200 #Kelvin


class MonteCarlo_2:


    def __init__(self, *args, transition_method='delta', sample_size=1000, delta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert transition_method in ['delta','uniform'], 'Unknown transition method'
        self.transition_method = transition_method
        self.delta = delta
    
    def direct_integration(self, property=None):
        if property is None:
            property = self.calc.potential_energy
        numerator_inner = lambda x: property(x) * np.exp(-self.calc.potential_energy(x) / self.kT)
        denominator_inner = lambda x: np.exp(-self.calc.potential_energy(x) / self.kT)
        numerator = quad(numerator_inner, self.xmin, self.xmax)[0] 
        denominator = quad(denominator_inner, self.xmin, self.xmax)[0] 
        return numerator / denominator
    
    def estimate_from_sample(self, property=None):
        if property is None:
            property = self.calc.potential_energy
        if self.sample is None:
            self.setup_sample()
        numerator = np.sum(property(self.sample))
        denominator = len(self.sample)
        return numerator / denominator
        
    def setup_sample(self):
        
        xs = []
        x = self.x
        for e in range(self.sample_size):
            xs.append(x)
            if self.transition_method == 'delta':
                x_new = x + self.delta*np.random.randn()
            else:
                x_new = self.xmin + (self.xmax-self.xmin)*np.random.rand()
            de = self.calc.potential_energy(x_new) - self.calc.potential_energy(x)
            if np.random.rand() < np.exp(-de/self.kT):
                x = x_new
        self.sample = np.array(xs)
        
        print('Sample size:',len(self.sample))
        
    def plot(self,ax,xwidth=0.1):
        super().plot(ax,xwidth)
        
        average_V_exact = self.direct_integration()
        average_V_sample = self.estimate_from_sample()
        ax.set_title(f'Ve={average_V_exact:.3f} Vmet-MC={average_V_sample:.3f}')
        ax.text(-1,1,f'kT={self.kT}')

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
        


def line_search(cluster,steps=100, tol = 0.05):
    
    test = cluster.copy()
    def energy_of_alpha(alpha,p):
        test.set_positions(cluster.get_positions() + alpha * p)
        return test.potential_energy
    
    for i in range(steps):
        f = cluster.forces()
        fnorm = np.linalg.norm(f)
        if fnorm < tol:
            break
        p = f/fnorm
        
        alpha_opt = fmin(lambda alpha: energy_of_alpha(alpha,p), 0.1, disp=False)
            
        cluster.set_positions(cluster.get_positions() + alpha_opt * p)



class ParticleSwarm():

    def __init__(self, cluster, num_particles, search_space, w = 1, c1 = 1, c2 = 1):
        self.cluster = cluster
        self.calc = cluster.calc
        self.inertia = w
        self.cognitive = c1
        self.social = c2

        self.positions = self.create_particles(num_particles, search_space)
        self.velocities = np.zeros((num_particles, 2))


        self.pbest_positions = self.positions.copy()
        self.pbest_values = self.evaluate_all(self.pbest_positions)

        gbest = np.argmin(self.pbest_values)

        self.gbest_position = self.positions[gbest]
        self.gbest_value = self.pbest_values[gbest]


        self.plot_artists = {}


    def create_particles(self,num_particles, search_space):
        # Unpack the search space boundaries
        (x_min, x_max), (y_min, y_max) = search_space

        # Generate random positions within the given boundaries
        x_positions = np.random.uniform(x_min, x_max, num_particles)
        y_positions = np.random.uniform(y_min, y_max, num_particles)

        # Combine x and y coordinates
        particles = np.array(list(zip(x_positions, y_positions)))

        return particles

    def plot_particles(self, ax, force_draw = False):
    
        # Extract x and y coordinates from particles
        
        pos = self.positions
        colors = [f'C{i}' for i in range(len(pos))]


        if self.plot_artists.get(ax, None) is None:
            self.plot_artists[ax] = ax.scatter(pos[:,0], pos[:,1], marker='o',c = colors, alpha = 0.5)
            self.plot_artists['gbest'] = ax.scatter(self.gbest_position[0], self.gbest_position[1], label='Global Best', marker='x', c='aqua')

        else:
            self.plot_artists[ax].set_offsets(pos)
            self.plot_artists['gbest'].set_offsets(self.gbest_position)

        # Set labels and title
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Particle Swarm Optimization')
        ax.legend(loc='upper right')



    def evaluate_all(self, positions):
        return np.array([self.evaluate(p) for p in positions])
    
    def evaluate(self, particle):
        cluster_positions = self.cluster.get_positions()
        particle_position = particle

        positions = np.array([particle_position, *cluster_positions])

        # Calculate the energy of the particle at the given position
        energy = self.calc.energy(positions)
        return energy


    def propagate(self):
        first_term = self.inertia * self.velocities.copy()
        second_term = self.cognitive * np.random.rand() * (self.pbest_positions - self.positions)
        third_term = self.social * np.random.rand() * (self.gbest_position - self.positions)
        self.velocities = first_term + second_term + third_term

        self.positions += self.velocities
        return self.positions, self.velocities

    def update(self):
        # Evaluate all particles
        values = self.evaluate_all(self.positions)
        if np.min(values) < self.gbest_value:
            self.gbest_value = np.min(values)
            self.gbest_position = self.positions[np.argmin(values)].copy()

        # Update personal bests
        mask = values < self.pbest_values
        self.pbest_values[mask] = values[mask]
        self.pbest_positions[mask] = self.positions[mask]


    def run(self, num_iterations):
        for i in range(num_iterations):
            self.propagate()
            self.update()
            print(f'Iteration {i+1}/{num_iterations} - Best value: {self.gbest_value} - Best position: {self.gbest_position}')


        


    





        
        
        


    


