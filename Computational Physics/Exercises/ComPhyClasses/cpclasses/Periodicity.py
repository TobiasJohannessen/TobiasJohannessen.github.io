
from .MolecularDynamics import StaticAtomicCluster, AtomicCluster
from .Potentials import Potential
import numpy as np

class PeriodicSystem(AtomicCluster):
    def __init__(self, *args, box = 10*np.eye(2), **kwargs):
        AtomicCluster.__init__(self, *args, **kwargs)
        self.box = box

    def copy(self):
        return PeriodicSystem(self.calc, pos = self.pos.copy(), static = self.static.copy(),box = self.box, kT = self.kT)



    @property
    def potential_energy(self):
        return self.calc.energy(self.pos, self.box)
    
    @property
    def volume(self):
        return np.linalg.det(self.box)

    def forces(self):
        forces = self.calc.forces(self.pos, self.box)
        return forces

    def stress(self, delta=1e-4):
        return self.calc.stress(self.pos, self.box, delta)
    
    def pressure(self):
        return self.calc.pressure(self.pos, self.box)


    def randomize_positions(self, scale = 0.1):
        pos = np.random.rand(self.N,2)
        pos[:,0] *= self.box[0,0]
        pos[:,1] *= self.box[1,1]
        self.set_positions(pos)


    def set_positions(self, pos):
        pos = pos.copy()
        pos[:,0] = pos[:,0] % self.box[0,0]
        pos[:,1] = pos[:,1] % self.box[1,1]
        self.pos = pos

    def get_scaled_positions(self):
        return self.pos @ np.linalg.inv(self.box).T

    def draw_periodic(self, ax, size=200, color = 'C0'):
        ax.scatter(self.pos[:,0], self.pos[:,1], s=size, c=color)
        for i, (x,y) in enumerate(self.pos):
            ax.text(x,y, str(i), fontsize=12, color='black', va='center', ha='center')

        deltas = [(0,0), (0,1), (1,0), (1,1), (-1,0), (0,-1), (-1,-1), (1,-1), (-1,1)]
        for dx,dy in deltas:
            ax.scatter(self.pos[:,0]+dx*self.box[0,0], self.pos[:,1]+dy*self.box[1,1], s=size, facecolor='w', edgecolor = color, alpha=0.5)

    def draw_cell(self, ax):
        cell_x = np.array([0, self.box[0,0], self.box[0,0], 0, 0])
        cell_y = np.array([0, 0, self.box[1,1], self.box[1,1], 0])
        ax.plot(cell_x, cell_y, 'k-')

    def title(self, ax):
        ax.set_title(r'$E_{pot} = ' + f'{self.potential_energy:.2f}$' \
                     + '\n' + r'$P = ' + f'{self.pressure():.2f}$' + '\n' + r'$Vol = ' + f'{self.volume:.2f}$')




def barostat(system):
    
    scaled_pos = system.get_scaled_positions()
    pos = system.get_positions()
    box = system.box
    stress = system.calc.stress(pos, box, delta=1e-3)
    alpha = 0.1
    scale_factor_x = 1 + alpha * stress[0,0]
    scale_factor_y = 1 + alpha * stress[1,1]

    minval = 0.999
    maxval = 1.001

    scale_factor_x = max(minval, min(maxval, scale_factor_x))
    scale_factor_y = max(minval, min(maxval, scale_factor_y))

    system.box[0,0] *= scale_factor_x
    system.box[1,1] *= scale_factor_y
    pos = scaled_pos @ system.box.T
    system.set_positions(pos)
    return stress
        
    

