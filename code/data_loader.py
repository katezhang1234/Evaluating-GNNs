import os.path as osp
import warnings
warnings.filterwarnings("ignore")

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array

import random
import numpy as np
import scipy.sparse as sp

from collections import deque

'''
Note: This code refers to UPFD code, the link is https://github.com/safe-graph/GNN-FakeNews
'''

"""
	Functions to help load the graph data
"""

def read_file(folder, name, dtype=None):
	path = osp.join(folder, '{}.txt'.format(name))
	return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
	"""
	PyG util code to create graph batches
	"""

	node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
	node_slice = torch.cat([torch.tensor([0]), node_slice])

	row, _ = data.edge_index
	edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
	edge_slice = torch.cat([torch.tensor([0]), edge_slice])

	# Edge indices should start at zero for every graph.
	data.edge_index -= node_slice[batch[row]].unsqueeze(0)
	data.__num_nodes__ = torch.bincount(batch).tolist()

	slices = {'edge_index': edge_slice}
	if data.x is not None:
		slices['x'] = node_slice
	if data.edge_attr is not None:
		slices['edge_attr'] = edge_slice
	if data.y is not None:
		if data.y.size(0) == batch.size(0):
			slices['y'] = node_slice
		else:
			slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

	return data, slices


def filter_graph_by_distance(data, max_distance):
    from collections import deque

    edge_index = data.edge_index.numpy()
    num_nodes = data.x.size(0)

    distances = [float('inf')] * num_nodes
    news_node = 0 
    distances[news_node] = 0

    queue = deque([news_node])	# BFS
    while queue:
        current = queue.popleft()
        for neighbor in edge_index[1, edge_index[0] == current]:
            if distances[neighbor] == float('inf'):  # Unvisited
                distances[neighbor] = distances[current] + 1
                if distances[neighbor] <= max_distance:
                    queue.append(neighbor)

    # Filter nodes and edges
    valid_nodes = [i for i, dist in enumerate(distances) if dist <= max_distance]
    node_mapping = {node: idx for idx, node in enumerate(valid_nodes)}
    valid_edges = [
        (node_mapping[u], node_mapping[v])
        for u, v in zip(edge_index[0], edge_index[1])
        if u in node_mapping and v in node_mapping
    ]
    valid_edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

    # Update data object
    data.x = data.x[valid_nodes]
    data.edge_index = valid_edge_index
    data.num_nodes = len(valid_nodes)  # Update node count

    return data



def read_graph_data(folder, feature, max_distance=2):
    """
    PyG util code to create PyG data instance from raw graph data
    """

    # Load data
    node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
    edge_index = read_file(folder, 'A', torch.long).t()
    node_graph_id = np.load(folder + 'node_graph_id.npy')
    graph_labels = np.load(folder + 'graph_labels.npy')

    edge_attr = None
    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    # Create PyG data object
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Filter each graph to a max distance
    data_list = []
    for graph_id in torch.unique(node_graph_id):
        # Extract subgraph for each graph ID
        mask = node_graph_id == graph_id
        subgraph = Data(
            x=data.x[mask],
            edge_index=data.edge_index[:, mask[node_graph_id[data.edge_index[0]]]],
            y=data.y[graph_id],
        )
        # Filter graph
        filtered_graph = filter_graph_by_distance(subgraph, max_distance)
        data_list.append(filtered_graph)

    # Rebuild slices
    data, slices = InMemoryDataset.collate(data_list)
    print("slices = ", slices)
    print("data = ", data)
    return data, slices


class ToUndirected:
	def __init__(self):
		"""
		PyG util code to transform the graph to the undirected graph
		"""
		pass

	def __call__(self, data):
		edge_attr = None
		edge_index = to_undirected(data.edge_index, data.x.size(0))
		num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
		# edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
		edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
		data.edge_index = edge_index
		data.edge_attr = edge_attr
		return data


class DropEdge:
	def __init__(self, tddroprate, budroprate):
		"""
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
		self.tddroprate = tddroprate
		self.budroprate = budroprate

	def __call__(self, data):
		edge_index = data.edge_index

		if self.tddroprate > 0:
			row = list(edge_index[0])
			col = list(edge_index[1])
			length = len(row)
			poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
			poslist = sorted(poslist)
			row = list(np.array(row)[poslist])
			col = list(np.array(col)[poslist])
			new_edgeindex = [row, col]
		else:
			new_edgeindex = edge_index

		burow = list(edge_index[1])
		bucol = list(edge_index[0])
		if self.budroprate > 0:
			length = len(burow)
			poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
			poslist = sorted(poslist)
			row = list(np.array(burow)[poslist])
			col = list(np.array(bucol)[poslist])
			bunew_edgeindex = [row, col]
		else:
			bunew_edgeindex = [burow, bucol]

		data.edge_index = torch.LongTensor(new_edgeindex)
		data.BU_edge_index = torch.LongTensor(bunew_edgeindex)
		data.root = torch.FloatTensor(data.x[0])
		data.root_index = torch.LongTensor([0])

		return data


class FNNDataset(InMemoryDataset):
	r"""
		The Graph datasets built upon FakeNewsNet data

	Args:
		root (string): Root directory where the dataset should be saved.
		name (string): The `name
			<https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
			dataset.
		transform (callable, optional): A function/transform that takes in an
			:obj:`torch_geometric.data.Data` object and returns a transformed
			version. The data object will be transformed before every access.
			(default: :obj:`None`)
		pre_transform (callable, optional): A function/transform that takes in
			an :obj:`torch_geometric.data.Data` object and returns a
			transformed version. The data object will be transformed before
			being saved to disk. (default: :obj:`None`)
		pre_filter (callable, optional): A function that takes in an
			:obj:`torch_geometric.data.Data` object and returns a boolean
			value, indicating whether the data object should be included in the
			final dataset. (default: :obj:`None`)
	"""

	def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
		self.name = name
		self.root = root
		self.feature = feature
		super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
		if not empty:
			self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_dir(self):
		name = 'raw/'
		return osp.join(self.root, self.name, name)

	@property
	def processed_dir(self):
		name = 'processed/'
		return osp.join(self.root, self.name, name)

	@property
	def num_node_attributes(self):
		if self.data.x is None:
			return 0
		return self.data.x.size(1)

	@property
	def raw_file_names(self):
		names = ['node_graph_id', 'graph_labels']
		return ['{}.npy'.format(name) for name in names]

	@property
	def processed_file_names(self):
		if self.pre_filter is None:
			# TODO: change this back
			# return f'{self.name[:3]}_data_max2_{self.feature}.pt'
			return f'{self.name[:3]}_data_{self.feature}.pt'
		else:
			return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

	def download(self):
		raise NotImplementedError('Must indicate valid location of raw data. No download allowed')
	
	def process(self):
		self.data, self.slices = read_graph_data(self.raw_dir, self.feature, max_distance=2)

		if self.pre_filter is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [data for data in data_list if self.pre_filter(data)]
			self.data, self.slices = self.collate(data_list)

		if self.pre_transform is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [self.pre_transform(data) for data in data_list]
			self.data, self.slices = self.collate(data_list)

		torch.save((self.data, self.slices), self.processed_paths[0])

	def __repr__(self):
		return '{}({})'.format(self.name, len(self))