import numpy as np
from scipy.integrate import quad
from scipy.spatial.distance import pdist,squareform
from scipy.optimize import fmin
from matplotlib.colors import to_rgba

def relax(cluster,steps=100):
    
    test = cluster.copy()
    def energy_of_alpha(alpha,p):
        test.set_positions(cluster.get_positions() + alpha * p)
        return test.potential_energy
    
    for i in range(steps):
        f = cluster.forces()
        fnorm = np.linalg.norm(f)
        if fnorm < 0.05:
            break
        p = f/fnorm
        
        alpha_opt = fmin(lambda alpha: energy_of_alpha(alpha,p), 0.1, disp=False)
            
        cluster.set_positions(cluster.get_positions() + alpha_opt * p)

def velocity_verlet(cluster, N=100, dt=0.01):
    for _ in range(N):
        r = cluster.get_positions()
        v = cluster.get_velocities()
        a_t = cluster.forces()
        r += v * dt + 0.5 * a_t * dt**2
        cluster.set_positions(r)
        a_t_dt = cluster.forces()
        v += 0.5 * (a_t + a_t_dt) * dt
        cluster.set_velocities(v)

class NonPeriodicSystem():
    
    def __init__(self, calc, N=None, pos=None, static=None,
                descriptor_method=None, kT=0.15):
        self.calc = calc
        assert (N is not None and pos is None) or \
               (N is None and pos is not None), 'You must specify either N or pos'
        if pos is not None:
            self.set_positions(np.array(pos)*1.)
            self.N = len(pos)
        else:
            self.N = N
            self.randomize_positions()
        if static is not None:
            assert len(static) == self.N, 'static must be N long'
            self.static = static
        else:
            self.static = [False for _ in range(self.N)]
        self.filter = np.array([self.static,self.static]).T
        self.indices_dynamic_atoms = \
                            [i for i,static in enumerate(self.static) if not static]
        self.plot_artists = {}
        self.descriptor_method = descriptor_method
        self.kT = kT
        self.velocities = np.zeros(self.pos.shape)
        
    def copy(self):
        return NonPeriodicSystem(self.calc, pos=self.pos.copy(), static=self.static.copy(),
                              kT=self.kT)

    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(self.velocities**2)

    def set_velocities(self,velocities):
        self.velocities = np.where(self.filter,0,velocities)

    def get_velocities(self):
        return self.velocities.copy()
    
    def energy_title(self):
        return f'Ek={self.kinetic_energy:.1f} ' +\
               f'Ep={self.potential_energy:.1f} ' + \
               f'E={self.potential_energy + self.kinetic_energy:.1f}'
    
    @property
    def potential_energy(self):
        return self.calc.energy(self.pos)
        
    def forces(self):
        forces = self.calc.forces(self.pos)
        return np.where(self.filter,0,forces)

    def randomize_positions(self):
        pos = np.random.rand(self.N,2)
        pos[:,0] *= 4
        pos[:,1] *= 4
        self.set_positions(pos)

    def set_positions(self,pos):
        self.pos = pos
        
    def get_positions(self):
        return self.pos.copy()
    
    @property
    def descriptor(self):
        return self.descriptor_method.descriptor(self.pos)

    def energy_title(self):
        return f'Ep={self.potential_energy:.1f}'
    
    def draw(self,ax,size=100,alpha=1,force_draw=False,edge=False,color='C0',
             energy_title=True):
        if self.plot_artists.get(ax,None) is None or force_draw or edge:
            colors = ['C1' if s else color for s in self.static]
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

