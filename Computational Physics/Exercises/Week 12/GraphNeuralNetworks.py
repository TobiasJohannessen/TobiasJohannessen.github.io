import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation



def elements_for_random_graph(num_nodes, cutoff = 2.5, box_size=10):
    positions = []
    for i in range(num_nodes):
        new_position = torch.rand(1, 2) * box_size
        if len(positions) > 0:
            all_positions = torch.vstack(positions)
            while torch.any(torch.cdist(all_positions, new_position) < 0.75*cutoff) or \
            torch.all(torch.cdist(all_positions, new_position) > cutoff):
                new_position = torch.rand(1, 2) * box_size        
        positions.append(new_position)

    positions = torch.vstack(positions)

    edge_index = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            edge_index.append([i, j])

    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.int64).reshape(2, -1)
    x = torch.tensor(list(range(num_nodes)),dtype=torch.float32)
    return  positions, edge_index, x

def random_graph(num_nodes, cutoff = 2.5, box_size=10):
    positions, edge_index, x = elements_for_random_graph(num_nodes, cutoff, box_size)
    edge_index = keep_short_edges(edge_index, positions)
    graph =  Data(pos=positions, edge_index=edge_index, x=x.view(-1, 1))
    return graph



def draw_graph(graph, ax, show_labels=True, color_style='default', feature_range =[0,10]):
    # Get or generate positions
    if graph.pos is not None:
        pos = graph.pos
    else:
        pos = torch.rand(graph.num_nodes, 2) * 5
        graph.pos = pos

    if hasattr(graph, 'xs'):
        numbers = graph.xs.detach().numpy()
    else:
        numbers = graph.x.detach().numpy()

    # Center positions around (0, 0)
    center_of_mass = torch.mean(pos, axis=0)
    pos = pos - center_of_mass

    # Edge and node data
    edge_index = graph.edge_index

    if color_style == 'default':
        colors = [f'C{int(np.abs((np.round(number))))}' for number in numbers]
         # Plot nodes
        ax.scatter(pos[:, 0], pos[:, 1], zorder=2, s=1000, edgecolors='black', c=colors)
    elif color_style == 'gradient':
        norm = plt.Normalize(vmin=feature_range[0], vmax=feature_range[1])
    
        ax.scatter(pos[:, 0], pos[:, 1], zorder=2, s=1000, edgecolors='black', c=numbers, cmap='jet', norm=norm)

    else:
        raise ValueError('Unknown color style')


   

    # Plot edges
    for src, dst in edge_index.T:
        ax.plot([pos[src, 0], pos[dst, 0]], [pos[src, 1], pos[dst, 1]], zorder=1, color='black', lw=2)

    # Add labels
    if show_labels:
        for i, number in enumerate(numbers):
            ax.text(pos[i, 0], pos[i, 1], f'{float(number):.1f}', fontsize=15, ha='center', va='center', zorder=3, color='white')
            ax.text(pos[i, 0] + 0.4, pos[i, 1] + 0.4, str(i), fontsize=8, ha='left', va='bottom', zorder=3,
                    color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))

    # Dynamic axis limits
    margin = 1
    x_min, x_max = pos[:, 0].min() - margin, pos[:, 0].max() + margin
    y_min, y_max = pos[:, 1].min() - margin, pos[:, 1].max() + margin

    x_min, y_min = min(x_min, y_min), min(x_min, y_min)
    x_max, y_max = max(x_max, y_max), max(x_max, y_max)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), aspect='equal')

    ax.set(xticks=[], yticks=[])



        

def check_identical(graph1, graph2):
    adj_matrix1 = to_dense_adj(graph1.edge_index).squeeze().numpy()
    adj_matrix2 = to_dense_adj(graph2.edge_index).squeeze().numpy()
  

    eigenvalues1 = sorted(np.linalg.eigvals(adj_matrix1))
    eigenvalues2 = sorted(np.linalg.eigvals(adj_matrix2))

    if np.allclose(eigenvalues1, eigenvalues2):
        return True
    else:
        return False


def keep_short_edges(edge_index,positions):
    edge_index_to_keep = []
    for edge in edge_index.T:
        source = positions[edge[0]]
        target = positions[edge[1]]
        distance = torch.dist(source, target)
        if distance < 2.5:
            edge_index_to_keep.append(edge)
    return torch.tensor(np.array(edge_index_to_keep).T)


from torch_geometric.nn.aggr import SumAggregation

class NoParamsGNN(MessagePassing):
    def __init__(self,aggr='add'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j): 
        return x_j

class PsiGNN(MessagePassing):

    def __init__(self, d_in, d_out, aggr = 'add'):
        super().__init__(aggr = aggr)
        self.psi = torch.nn.Linear(d_in, d_out)

    def forward(self, x, edge_index):
        x = x.view(-1, 1)
        x = self.propagate(edge_index, x = x)
        x = x.flatten()
        return x

    def message(self, x_j):
        return self.psi(x_j)

class PhiPsiGNN(MessagePassing):

    def __init__(self, d_in, d_out, aggr = 'add'):
        super().__init__(aggr = aggr)
        self.psi = torch.nn.Linear(d_in, d_out)
        self.phi = torch.nn.Linear(d_in + d_out, d_out)

    def forward(self, x, edge_index):
        #x = x.view(-1, 1)
        psi_of_x_j = self.propagate(edge_index, x = x)
        x = self.phi(torch.cat([x, psi_of_x_j], dim = 1))
        x = x.flatten()
        return x

    def message(self, x_j):
        return self.psi(x_j)


  
class AggrPhiPsiGNN(MessagePassing):
    def __init__(self, d_in, d_out, aggr = 'add'):
        super().__init__(aggr = aggr)
        self.psi = torch.nn.Linear(d_in, d_out)
        self.phi = torch.nn.Linear(d_in + d_out, d_out)
        self.aggr = SumAggregation()

    def forward(self, x, edge_index, batch):
        x = x.view(-1, 1)
        psi_of_x_j = self.propagate(edge_index, x = x)
        x = self.phi(torch.cat([x, psi_of_x_j], dim = 1))
        y = self.aggr(x, batch)
        x = x.flatten()
        return x,y # return both local and global prediction

    def message(self, x_j):
        return self.psi(x_j)


