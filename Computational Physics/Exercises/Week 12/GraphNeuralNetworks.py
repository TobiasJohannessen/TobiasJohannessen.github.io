import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation



def elements_for_random_graph(num_nodes):
    cutoff = 2.5

    box_size=10
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



def draw_graph(graph, ax):
    if graph.pos is not None:
        pos = graph.pos
        
    else:
        pos = torch.rand(graph.num_nodes, 2) * 5
        graph.pos = pos

    center_of_mass = torch.mean(pos, axis = 0)
    pos = pos - center_of_mass
    edge_index = graph.edge_index
    numbers = graph.x
    
   
    ax.scatter(pos[:, 0], pos[:, 1], zorder = 2, s = 1000, edgecolors = 'black')

    
    for src, dst in edge_index.T:
        ax.plot([pos[src, 0], pos[dst, 0]], [pos[src, 1], pos[dst, 1]], zorder = 1, color = 'black', lw = 2)
    
    for i, number in enumerate(numbers):
       ax.text(pos[int(i),0], pos[int(i),1], str(int(number)), fontsize=15, ha='center', va = 'center', zorder = 3, color='white')
       ax.text(pos[int(i),0] + 0.15 , pos[int(i),1] + 0.15, str(i), fontsize=8, ha='left', va = 'bottom', zorder = 3, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
    

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

