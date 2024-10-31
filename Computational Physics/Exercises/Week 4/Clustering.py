from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.spatial.distance import pdist,squareform

############################################################################################################

# KMeans class for clustering

############################################################################################################

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.clusters = None
        self.labels = None

    def fit(self, X):    

        #Pick out some random points in the data to be the initial centroids
        M, n_features = X.shape
        random_indices = np.random.choice(M, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        
        for i in range(self.max_iter):

            
            #Calculate the distance between each point and each centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            #Find which centroid is closest to each point. This is equal to the cluster assignment
            self.labels = np.argmin(distances, axis=1)
            
            #Calculate the new centroids as the mean of all points assigned to each cluster
            new_centroids= np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])


            #If the centroids have not moved much, we have converged
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f'Converged after {i} iterations')
                
                break
            self.centroids = new_centroids  
            

       

        return self.centroids



    def predict(self, X):        
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        
        self.labels = np.argmin(distances, axis=1)
        return self.labels


############################################################################################################

# PRINCIPAL COMPONENT ANALYSIS

############################################################################################################



class PCA():

    def __init__(self, n_components = 2, color='C0'):
        self.n_components = n_components
        self.desc_color = color
        

    def fit(self, X):
        X_centered = X - np.mean(X,axis=0)
        

        cov_matrix = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]


        Z_full = X_centered @ eigenvectors

        return Z_full
    
    def transform(self, X):
        Z_k = self.fit(X)[:,:self.n_components]

        return Z_k


    def draw_descriptor(self, pos, ax):
        Z_k = self.transform(pos)
        ax.plot(Z_k[:,0],Z_k[:,1],'o',color=self.desc_color, alpha = 0.5, ms= 10)
        ax.set_title(self.__class__.__name__)
        ax.set(xlabel='PC1',ylabel='PC2')




############################################################################################################

# COULOMB MATRIX SPECTRUM

############################################################################################################



class CoulombMatrixSpectrum():
    
    def __init__(self, color='C4'):
        self.xwidth = 1
        self.desc_color = color
    
    def descriptor(self, pos):
        # Calculate the connectivity matrix where 1 means that two particles are within the cutoff distance and 0 means they are not
        connectivity_matrix = (squareform(1/pdist(pos)))

        # Set the diagonal to zero, as atoms cannot be their own nearest neighbor
        np.fill_diagonal(connectivity_matrix,1)
        
        # Calculate the eigenvalues of the connectivity matrix
        eigenvalues = np.linalg.eigvals(connectivity_matrix)

        return sorted(eigenvalues)

    def draw_descriptor(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.desc_color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([-2,8])
        ax.set_title(self.__class__.__name__)


############################################################################################################

# CONNECTIVITY GRAPH SPECTRUM

############################################################################################################


class ConnectivityGraphSpectrum():
    
    def __init__(self, color='C3', sigma = 2**(-1/6)):
    
        self.xwidth = 1
        self.desc_color = color
        self._r_min = 2**(1/6) * sigma
        self._A = 1.2
        self.r_cut = self._A * self._r_min
    
    def descriptor(self,pos):
        # Calculate the connectivity matrix where 1 means that two particles are within the cutoff distance and 0 means they are not
        connectivity_matrix = (squareform(pdist(pos)) < self.r_cut).astype(int)
        # Set the diagonal to zero, as atoms cannot be their own nearest neighbor
        np.fill_diagonal(connectivity_matrix,0)
        
        # Calculate the eigenvalues of the connectivity matrix
        eigenvalues = np.linalg.eigvals(connectivity_matrix)

        return sorted(eigenvalues)
    
    def draw_descriptor(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.desc_color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)


############################################################################################################

# COORDINATION NUMBERS

############################################################################################################

class CoordinationNumbers():
    
    def __init__(self, color='C2', sigma = 2**(-1/6)):
        self.xwidth = 1
        self.desc_color = color
        self._r_min = 2**(1/6) * sigma
        self._A = 1.2
        self.r_cut = self._A * self._r_min
    
    def descriptor(self,pos):
        # Calculate the connectivity matrix where 1 means that two particles are within the cutoff distance and 0 means they are not
        connectivity_matrix = (squareform(pdist(pos)) < self.r_cut).astype(int)

        # Set the diagonal to zero, as atoms cannot be their own nearest neighbor
        np.fill_diagonal(connectivity_matrix,0)

        sums = connectivity_matrix.sum(axis=1)
        return sums
    
    def draw_descriptor(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.desc_color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)


############################################################################################################

# PAIR DISTANCES

############################################################################################################

class PairDistances():
    
    def __init__(self, color='C1'):
        self.xwidth = 0.5
        self.desc_color = color
        self.bin_edges = np.arange(0,7.01,self.xwidth)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) /2
    
    def descriptor(self,pos):
        distances = pdist(pos)
        hist, _ = np.histogram(distances,bins=self.bin_edges)
        return hist
    
    def draw_descriptor(self,pos,ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.desc_color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)


############################################################################################################

# EXTREME NEIGHBOR COUNT

############################################################################################################

class ExtremeNeighborCount():
    
    def __init__(self, color='C5', sigma = 2**(-1/6)):
        self.xwidth = 1
        self.desc_color = color
        self._color = None
        self.bin_centers = range(2)
        self.name = 'ExtremeNeighborCount'
        self._r_min = 2**(1/6) * sigma
        self._A = 1.2
        self.r_cut = self._A * self._r_min
        self._cluster = 0
        self._N_lowest = None
        self._N_highest = None


    
    def cluster(self, pos):
        permutations = [[2,4], [1,5], [2,5], [1,6], [2,6], [3,6]]

        for i,permutation in enumerate(permutations):
            if (self.N_lowest(pos) == permutation[0]) and (self.N_highest(pos) == permutation[1]):
                self._cluster = i+1
                break
            else:
                self._cluster = 0

        return self._cluster
    

    def color(self, pos):
        self._color = f'C{self.cluster(pos)}'
        return self._color

    def N_lowest(self, pos):
        if self._N_lowest is None:
            self._N_lowest = self.descriptor(pos)[0]
        return self._N_lowest

    def N_highest(self, pos):
        if self._N_highest is None:
            self._N_highest = self.descriptor(pos)[1]
        return self._N_highest

    
    def descriptor(self,pos):
        # Calculate the connectivity matrix where 1 means that two particles are within the cutoff distance and 0 means they are not
        connectivity_matrix = (squareform(pdist(pos)) < self.r_cut).astype(int)

        # Set the diagonal to zero, as atoms cannot be their own nearest neighbor
        np.fill_diagonal(connectivity_matrix,0)

        sums = connectivity_matrix.sum(axis=1)
        N_lowest = np.min(connectivity_matrix.sum(axis=1))
        N_highest = np.max(connectivity_matrix.sum(axis=1))
        
        return np.array([N_lowest,N_highest])

    def draw_descriptor(self, pos, ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector, width=0.8 * self.xwidth,color=self.desc_color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,7])
        xticklabels = ['$N_{lowest}$','$N_{highest}$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)


    def draw_cluster(self, pos, ax):
        vector = self.descriptor(pos)
        plot_element = ax.plot(vector[0], vector[1],'o', color=self.color(pos), alpha = 0.5, ms= 10)
        ax.set_title(self.__class__.__name__)
        return plot_element


############################################################################################################

# DISTANCE MOMENTS

############################################################################################################

class DistanceMoments():
    
    def __init__(self,color='C4'):
        self.xwidth = 1
        self.bin_centers = range(2)
        self.name = 'DistanceMoments'
        self._cluster = None
        self._color = None
        self.desc_color = color


    
    def color(self, pos):
        self._color = f'C{self.cluster(pos)}'
        return self._color
        
    
    def descriptor(self, pos):
        all_distances = pdist(pos)
        mean = np.mean(all_distances)
        std = np.std(all_distances)
        return np.array([mean,std])

    
    def cluster(self, pos):
        borders = [1.60,1.695,1.8,1.85,1.9,2.1]
        mean = self.descriptor(pos)[0]
        for i in range(len(borders) - 1):
            if (mean >= borders[i]) and (mean < borders[i+1]):
                self._cluster = i+1
                break
            else:
                self._cluster = len(borders)
        return self._cluster
    
    def draw_descriptor(self,pos,ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.desc_color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,2.3])
        xticklabels = [r'$\mu$',r'$\sigma$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)

    def draw_cluster(self, pos, ax):
        mean, std = self.descriptor(pos)

        plot_element = ax.plot(mean, std,'o', color=self.color(pos), alpha = 0.5, ms= 10)
        ax.text(mean, std, f'{self.cluster(pos)}', fontsize=12, color=self.color(pos))
        ax.set_title(self.__class__.__name__)
        return plot_element



############################################################################################################

# ATOMIC CLUSTER CLASS

############################################################################################################


class AtomicCluster():

    def __init__(self, positions, descriptor_methods = None):
        self.pos = positions
        if hasattr(descriptor_methods, '__iter__') == False:
            descriptor_methods = [descriptor_methods]
        self.descriptor_methods = [descriptor() for descriptor in descriptor_methods]
        self._cluster = None
        self._color = None
        


    #Calculate a given descriptor for the atomic cluster
    def get_descriptor(self, method):
        return method().descriptor(self.pos)
    
    

    def cluster_color(self, descriptor):
        if hasattr(descriptor, 'cluster'):
            clustercolor = f'C{descriptor.cluster(self.pos)}'
        else:
            clustercolor = 'C0'
        return clustercolor



    #Functions to draw the atomic cluster and its descriptors

    def draw(self, ax, offset = [0,0], ms = 30, descriptor = False, set_color = None, alpha = 0.5, alpha_edge = 1):
        
        if set_color is not None:
            self._color = set_color
        else:
            if descriptor is False:
                descriptor = self.descriptor_methods[0]
            self._color = self.cluster_color(descriptor)

            
        for pos in self.pos:
            circle = mpatches.Circle(pos + offset, 0.5, facecolor=self._color, alpha = alpha)
            circle_edge = mpatches.Circle(pos + offset, 0.5, facecolor='none', edgecolor='black', lw=1, alpha = alpha_edge)
            ax.add_patch(circle)
            ax.add_patch(circle_edge)
        
        ax.set_aspect('equal')
        ax.set_title('Atomic Cluster')
        ax.set(xlim = [-4,4], ylim = [-4,4])
        return ax


    def draw_descriptor(self, ax):
        if self.descriptor_methods is not None:
            for descriptor in self.descriptor_methods:
                descriptor.draw_descriptor(self.pos, ax)

    def draw_descriptors(self, axs):
        if self.descriptor_methods is not None:
            if hasattr(axs, '__iter__') == False:
                axs = [axs]
            assert len(axs) == len(self.descriptor_methods), 'Number of axes must match number of descriptors'
            for ax, descriptor in zip(axs, self.descriptor_methods):
                self.draw_descriptor(ax, descriptor)

    def draw_cluster(self, axs):
        if self.descriptor_methods is not None:
            #assert len(axs) == len(self.descriptor_methods), 'Number of axes must match number of descriptors'
            plot_elements = []
            if hasattr(axs, '__iter__') == False:
                axs = [axs]
            for ax, descriptor in zip(axs, self.descriptor_methods):
                plot_elements.append(descriptor.draw_cluster(self.pos, ax))
            return plot_elements
        
    def draw_all(self, plot_color = None):
        if self.descriptor_methods is not None:
            descriptors = self.descriptor_methods
            fig, axs = plt.subplots(1, len(descriptors) + 1, figsize=(len(descriptors)*4, 2.5))
            #fig, axs = plt.subplots(2,3, figsize = (12, 8))
            #axs = axs.flatten()
            axes = np.zeros(len(descriptors) + 1).astype(object)
            axes[0] = self.draw(axs[0], set_color = plot_color)
            for ax, descriptor, i in zip(axs[1:], descriptors, range(len(descriptors))):
                axes[i + 1] = descriptor.draw_descriptor(self.pos, ax)

            return fig, axes
        


