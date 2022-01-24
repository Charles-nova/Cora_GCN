import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

# class MyOwnDataset(InMemoryDataset):


x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

'''the graph connectivity(edge index) should be confined with the COO format
   the first lists contains the index of the source nodes, while the index of target nodes is 
   specified in the second list. edge_index` needs to be of type `torch.long`'''
edge_index = torch.tensor([[0, 1, 2, 3, 0],
                           [1, 0, 1, 2, 3]], dtype=torch.long)
example = Data(edge_index=edge_index, x=x)
print(example.x)