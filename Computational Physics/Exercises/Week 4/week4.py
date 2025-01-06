import numpy as np
import matplotlib.pyplot as plt
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("axes", titlesize=16)
plt.rc("font", size=12)
from scipy.spatial.distance import pdist,squareform
from scipy.optimize import fmin
from matplotlib.colors import to_rgba

class LennardJones():
    def __init__(self,eps0=5,sigma=2**(-1/6)):
        self.eps0 = eps0
        self.sigma = sigma
        
    def _V(self,r):
        return 4 * self.eps0 * ( (self.sigma/r)**12 - (self.sigma/r)**6 )

    def _dV_dr(self, r):
        return -4 * self.eps0 * (12 * (self.sigma / r)**12 - 6 * (self.sigma / r)**6) / r    

    def energy(self, pos):
        return np.sum(self._V(pdist(pos)))
    
    def forces(self, pos):
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(r, np.inf)
        force_magnitude = self._dV_dr(r)
        forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                        r[..., np.newaxis], axis=1)
        return forces

class AtomicCluster():

    def __init__(self, calc, N=None, pos=None, static=None, b=4,
                 descriptor_method=None):
        self.calc = calc
        assert (N is not None and pos is None) or \
               (N is None and pos is not None), 'You must specify either N or pos'
        if pos is not None:
            self.pos = np.array(pos)*1.
            self.N = len(pos)
        else:
            self.N = N
            self.pos = 2*self.b*np.random.rand(N,2) - self.b
        if static is not None:
            assert len(static) == self.N, 'static must be N long'
            self.static = static
        else:
            self.static = [False for _ in range(self.N)]
        self.filter = np.array([self.static,self.static]).T
        self.indices_dynamic_atoms = \
                            [i for i,static in enumerate(self.static) if not static]
        self.b = b
        self.plot_artists = {}
        self.descriptor_method = descriptor_method

    def copy(self):
        return AtomicCluster(self.calc, pos=self.pos, static=self.static, b=self.b)
        
    @property
    def energy(self):
        return self.calc.energy(self.pos)

    def forces(self):
        forces = self.calc.forces(self.pos)
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
            ax.set_title(f'E={self.energy:8.3f}')

    def draw_descriptor(self,ax):
        self.descriptor_method.draw(self.pos,ax)


class DistanceMoments():
    
    def __init__(self, color='C4'):
        self.xwidth = 1
        self.color = color
        self.bin_centers = range(2)
    
    def descriptor(self,pos):
        all_distances = pdist(pos)
        mean = np.mean(all_distances)
        std = np.std(all_distances)
        return np.array([mean,std])
    
    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,2.3])
        xticklabels = ['$\mu$','$\sigma$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)


class ExtremeNeighborCount():
    
    def __init__(self, color='C5'):
        self.xwidth = 1
        self.color = color
        self.bin_centers = range(2)
    
    def descriptor(self,pos):
        connectivity_matrix = (squareform(pdist(pos)) < 1.2).astype(int)
        np.fill_diagonal(connectivity_matrix, 0)
        coordination_numbers = np.sum(connectivity_matrix,axis=1)
        Nlowest = np.min(coordination_numbers)
        Nhighest = np.max(coordination_numbers)
        return np.array([Nlowest,Nhighest])

    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,7])
        xticklabels = ['$N_{lowest}$','$N_{highest}$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)


class PairDistances():
    
    def __init__(self, color='C1'):
        self.xwidth = 0.5
        self.color = color
        self.bin_edges = np.arange(0,7.01,self.xwidth)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) /2
    
    def descriptor(self,pos):
        bars, _ = np.histogram(pdist(pos),self.bin_edges)
        return bars
    
    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,20.1])
        ax.set_title(self.__class__.__name__)

class ConnectivityGraphSpectrum():
    
    def __init__(self, color='C3'):
        self.xwidth = 1
        self.color = color
    
    def descriptor(self,pos):
        connectivity_matrix = -(squareform(pdist(pos)) < 1.2).astype(int)
        np.fill_diagonal(connectivity_matrix, 0)
        eigen_values, _ = np.linalg.eig(connectivity_matrix)
        eigen_values = np.real(eigen_values) # ignore any small complex component
        sorted_eigen_values = sorted(eigen_values)
        return sorted_eigen_values

    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([-4.1,2.3])
        ax.set_title(self.__class__.__name__)

class CoordinationNumbers():
    
    def __init__(self, color='C2'):
        self.xwidth = 1
        self.color = color
    
    def descriptor(self,pos):
        connectivity_matrix = (squareform(pdist(pos)) < 1.2).astype(int)
        np.fill_diagonal(connectivity_matrix, 0)
        coordination_numbers = np.sum(connectivity_matrix,axis=1)
        xs = np.arange(0,8.01,self.xwidth)
        bars, _ = np.histogram(coordination_numbers,xs)
        return bars

    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,6.2])
        ax.set_title(self.__class__.__name__)

class CoulombMatrixSpectrum():
    
    def __init__(self, color='C4'):
        self.xwidth = 1
        self.color = color
    
    def descriptor(self,pos):
        r_matrix = squareform(pdist(pos))
        np.fill_diagonal(r_matrix, 1)
        one_over_r_matrix = r_matrix**-1
        eigen_values, _ = np.linalg.eig(one_over_r_matrix)
        eigen_values = np.real(eigen_values) # ignore any small complex component
        sorted_eigen_values = sorted(eigen_values)
        return sorted_eigen_values

    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([-2,8])
        ax.set_title(self.__class__.__name__)

