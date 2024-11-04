import numpy as np
from scipy.integrate import quad
from scipy.optimize import fmin
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt


#######################################################################
# A general class for systems that can be simulated i.e. have some sort of sampling.
#######################################################################

class SimulationSystem():
    def __init__(self, calc, m=1, x=0, kT=0.15, xmin=-3, xmax=3,
                 sample_size=10000):
        self.calc = calc
        self.m = m
        self.x = x
        self._kT = kT
        self.xmin = xmin
        self.xmax = xmax
        self.sample = None
        self.sample_size = sample_size
        
    @property
    def kT(self):
        return self._kT
    
    @kT.setter
    def kT(self,new_kT):
        if self._kT != new_kT:
            print('... changing kT')
            self.sample = None
            self._kT = new_kT
            
    def potential_energy(self):
        return self.calc.potential_energy(self.x)
    
    def get_position(self):
        return self.x
    
    def set_position(self,x):
        self.x = x
        
    def setup_sample(self):
        if self.sample is None:
            self.sample = []
        
    def plot(self,ax,xwidth=0.1):
        # plot potential
        xs = np.linspace(self.xmin, self.xmax, 100)
        ax.plot(xs, self.calc.potential_energy(xs))
        
        if self.sample is None:
             self.setup_sample()
        
        xs = np.arange(self.xmin,self.xmax,xwidth)
        bars, xs = np.histogram(self.sample,xs)
        
        xvals = (xs[:-1] + xs[1:]) / 2
        delta_xvals = (xvals[1] - xvals[0])
        P = bars / np.sum(bars) / delta_xvals
        width = delta_xvals * 0.8
        ax.bar(xvals, 0.5*P, width=width)
        
        vpotave = np.mean([self.calc.potential_energy(x) for x in self.sample])
        ax.set_title(f'<Vpot>={vpotave:.3f}')




########################################################################################
class MonteCarloSystem(SimulationSystem):
        
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

def velocity_verlet_1d(system, N=100):
    rs = []
    for _ in range(N):
        dt = 0.01
        r = system.get_position()
        v = system.get_velocity()
        a_t = system.force()
        r += v * dt + 0.5 * a_t * dt**2
        system.set_position(r)
        a_t_dt = system.force()
        v += 0.5 * (a_t + a_t_dt) * dt
        system.set_velocity(v)
        rs.append(r)
        
    return rs

class MolDynSystem(SimulationSystem):
    def __init__(self, *args, thermostat=None, verlet_steps=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = 0
        self.thermostat = thermostat
        self.verlet_steps = verlet_steps

    def force(self):
        return self.calc.force(self.x)

    def get_velocity(self):
        return self.v
    
    def set_velocity(self,v):
        self.v = v
        
    def setup_sample(self):
        if self.sample is None:
            self.sample = []
        for _ in range(self.sample_size):
            r = velocity_verlet_1d(self,N=self.verlet_steps)
            self.sample.append(r[-1])
            if self.thermostat is not None:
                self.thermostat(self)
           
        
    def plot(self,ax,xwidth=0.1):
        super().plot(ax,xwidth)
        
        vpotave = np.mean([self.calc.potential_energy(x) for x in self.sample])
        ax.set_title(f'<Vpot>={vpotave:.3f}')


def relax(cluster,steps=100, tol = 0.05):
    
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


############################################
# Class for atomic clusters that do not move dynamically
############################################

class StaticAtomicCluster():

    def __init__(self, calc, N=None, pos=None, static=None, b=4,
                descriptor_method=None, kT=0.15, periodicity= [0, 0], labels = None):
        self.b = b
        self.calc = calc
        assert (N is not None and pos is None) or \
               (N is None and pos is not None), 'You must specify either N or pos'
        if pos is not None:
            self.pos = np.array(pos)
            self.N = len(pos)
        else:
            self.N = N
            self.pos = 2*self.b*np.random.rand(N,2) - self.b
        if static is not None:
            assert len(static) == self.N, 'static must be N long'
            self.static = static
        else:
            self.static = [False for _ in range(self.N)]

        self.periodicity = periodicity
        if periodicity == [0, 0]:
            self.periodicity = False
        self.labels = None
        if labels is not None:
            self.labels = labels
        self.filter = np.array([self.static,self.static]).T
        self.indices_dynamic_atoms = \
                            [i for i,static in enumerate(self.static) if not static]
        
        self.plot_artists = {}
        self.descriptor_method = descriptor_method
        self.kT = kT

    def copy(self):
        return StaticAtomicCluster(self.calc, pos=self.pos, static=self.static, b=self.b)

    @property
    def potential_energy(self):
        return self.calc.energy(self.pos)

    def forces(self):
        forces = self.calc.force(self.pos)
        return np.where(self.filter,0,forces)

    def set_positions(self,pos,ignore_b=False):
        if ignore_b:
            self.pos = pos
        else:
            self.pos = (pos + self.b) % (2*self.b) - self.b

    def get_positions(self):
        return self.pos.copy()

    @property
    def descriptor(self):
        return self.descriptor_method.descriptor(self.pos)

    def energy_title(self):
        return f'Ep={self.potential_energy:.2g}'
    
    def draw(self,ax,size=100,alpha=1,force_draw=False,edge=False,color='C0',
             energy_title=True):
        if self.plot_artists.get(ax,None) is None or force_draw or edge:
            colors = ['C1' if s else color for s in self.static]
            if self.labels is not None:
                colors = [f'C{label}' for label in self.labels]
            facecolors = [to_rgba(c,alpha) for c in colors]
            if edge:
                edgecolors = (0,0,0,1)
                self.plot_artists[ax] = ax.scatter(self.pos[:,0],self.pos[:,1],
                                               s=size,facecolors=facecolors,
                                                  edgecolors=edgecolors,
                                                  linewidth=2)
            else:
                edgecolors = (0,0,0,0.5)
                self.plot_artists[ax] = ax.scatter(self.pos[:,0],self.pos[:,1],
                                                   s=size,facecolors=facecolors,
                                                  edgecolors=edgecolors,)
        else:
            self.plot_artists[ax].set_offsets(self.pos)
        if energy_title:
            ax.set_title(self.energy_title())

    def draw_descriptor(self,ax):
        self.descriptor_method.draw(self.pos,ax)



############################################
# Class for atomic clusters that move dynamically i.e. have velocities
############################################
class AtomicCluster(StaticAtomicCluster):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocities = np.zeros(self.pos.shape)
         
    def copy(self):
        return AtomicCluster(self.calc, pos=self.pos, static=self.static, b=self.b)
    
    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(self.velocities**2)

    def set_velocities(self,velocities):
        self.velocities = np.where(self.filter,0,velocities)

    def get_velocities(self):
        return self.velocities.copy()
    
    def energy_title(self):
        return f'Ek={self.kinetic_energy:.2g} ' +\
               f'Ep={self.potential_energy:.2g} ' + \
               f'E={self.potential_energy + self.kinetic_energy:.2g}'


def velocity_verlet(cluster, N = 100):
    for _ in range(N):
        dt = 0.01
        r = cluster.get_positions()
        v = cluster.get_velocities()
        a_t = cluster.forces()
        r += v * dt + 0.5 * a_t * dt**2
        cluster.set_positions(r)
        a_t_dt = cluster.forces()
        v += 0.5 * (a_t + a_t_dt) * dt
        cluster.set_velocities(v)
    