import os.path as osp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, add_remaining_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array

import random
import numpy as np
import scipy.sparse as sp

import pickle
import datetime
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


def read_graph_data(folder, feature):
	"""
	PyG util code to create PyG data instance from raw graph data
	"""
	if isinstance(feature, str):
		node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
	elif isinstance(feature, list): ## concat the multiple featues
		node_attributes = sp.hstack([sp.load_npz(folder + f'new_{f}_feature.npz') for f in feature])
	edge_index = read_file(folder, 'A', torch.long).t()
	node_graph_id = np.load(folder + 'node_graph_id.npy')
	graph_labels = np.load(folder + 'graph_labels.npy')


	edge_attr = None
	x = torch.from_numpy(node_attributes.todense()).to(torch.float)
	node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
	y = torch.from_numpy(graph_labels).to(torch.long)
	_, y = y.unique(sorted=True, return_inverse=True)

	num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
	edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
	edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
	data, slices = split(data, node_graph_id)

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
			#self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])
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
		if isinstance(self.feature, str):
			_feature = self.feature
		elif isinstance(self.feature, list): # for multiple features
			_feature = '_'.join(self.feature)
		if self.pre_filter is None:
			return f'{self.name[:3]}_data_{_feature}.pt'
		else:
			return f'{self.name[:3]}_data_{_feature}_prefilter.pt'

	# @property
	# def label(self):
	# 	return self.data[idx].y for idx in range(len(self))

	def download(self):
		raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

	def process(self):

		self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

		if self.pre_filter is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [data for data in data_list if self.pre_filter(data)]
			self.data, self.slices = self.collate(data_list)

		if self.pre_transform is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [self.pre_transform(data) for data in data_list]
			self.data, self.slices = self.collate(data_list)

		#self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
		#self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
		#self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)


		#torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])
		torch.save((self.data, self.slices), self.processed_paths[0])
	def __repr__(self):
		return '{}({})'.format(self.name, len(self))

class ConcatGraphDataset(Dataset):
	def __init__(self, *datasets):
		self.datasets = datasets
		self.concat_x = torch.cat((self.datasets[0].data.x, self.datasets[1].data.x), dim=1)
		node_graph_id = np.load(self.datasets[0].raw_dir + 'node_graph_id.npy')
		self.node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
#		if len(self.datasets) > 2:
#			for d in self.datasets[2:]:
#				self.concat_x = torch.cat((self.concat_x, d.x), dim=1)
	def __getitem__(self, i):
		self.data = Data(x=self.concat_x, edge_index=self.datasets[0][i].edge_index, edge_attr=self.datasets[0][i].edge_attr, y=self.datasets[0][i].y)
		self.dataset, _ = split(self.data, self.node_graph_id)

		return self.dataset
	def __len__(self):
		return min(len(d) for d in self.datasets)

	@property
	def num_features(self):
		return sum(d.num_features for d in self.datasets)

	@property
	def num_classes(self) -> int:
		return self.datasets[0].num_classes


class Custom_Hetero_Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset, node_type_names, edge_type_names):
		self.dataset = dataset
		self.node_type_names = node_type_names
		self.edge_type_names = edge_type_names
		super(Custom_Hetero_Dataset, self).__init__()

	def __getitem__ (self, idx):
		node_type_tensor = [1]*self.dataset[idx].num_nodes
		node_type_tensor[0] = 0
		each_edge = self.dataset[idx].edge_index
		edge1 = (each_edge[1]==0)*1 # tweets to news
		edge3 = ((each_edge[0]!=0)&(each_edge[1]!=0))*3 # tweets to tweets
		edge1[0] = 2 # news-self
		edge_type_tensor = edge1 + edge3
#		edge_type_tensor[(each_edge[1]==0)&(each_edge[0]!=0)] = 1
		#edge_type_tensor = tweeted_edge_tensor + retweet_edge_tensor
#		edge_type_tensor[each_edge[0] == each_edge[1]] = 2
		# edge_type_tensor[each_edge[0] == each_edge[1]] = 3 # tweets->tweets == 3
		# edge_type_tensor[0] = 2 # news->news ==2

		data = self.dataset[idx].to_heterogeneous(node_type=torch.tensor(node_type_tensor), edge_type=edge_type_tensor, node_type_names=self.node_type_names, edge_type_names=self.edge_type_names)
		return data
	
	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		for idx in range(len(self)):
			yield self.data[idx]
   
def make_temporal_weight(data, name):
	with open(f"data/{name[:3]}_id_time_mapping.pkl", 'rb') as f:
		time = pickle.load(f)
	time_dict = {}
	idx = 0
	for b_i, each_batch in enumerate(data):
		time_dict[b_i] = {0: 0}
		leng = each_batch['x'].shape[0]
		for g_i, each_idx in enumerate(range(idx+1, idx+leng), 1):
			time_dict[b_i][g_i] = datetime.datetime.fromtimestamp(time[each_idx])
		idx += leng
	return time_dict

def make_edge_weight(data, time_dict, unit='minute', use_depth_divide=False):
	edge_index_list, edge_weight_list, edge_attr_list = [], [], []
	DEPTH = 1
	for b_i, each_batch in enumerate(data):
		edge = remove_self_loops(each_batch.edge_index)[0]
		e1, e2 = edge
		each_weight = []
		for s, t in zip(e1, e2):
			s, t = s.item(), t.item()
			if s==t:
				each_weight.append(1)
			elif ((s==0) or (s==1)) & ((t==0) or (t==1)):
				each_weight.append(1)
			elif s==0:
				score = 1 + abs(time_dict[b_i][1] - time_dict[b_i][t]).total_seconds()
				if unit=='minute':
					score = score/60
				each_weight.append(score)
			elif t==0:
				score = 1 + abs(time_dict[b_i][s] - time_dict[b_i][1]).total_seconds()
				if unit=='minute':
					score = score/60
				each_weight.append(score)
			else:
				score = 1 + abs(time_dict[b_i][t] - time_dict[b_i][s]).total_seconds()
				if unit=='minute':
					score = score/60
				each_weight.append(())
		each_weight = 1/np.log1p(each_weight)
		edge, weight = add_remaining_self_loops(edge, torch.tensor(each_weight))
#        edge, weight = add_remaining_self_loops(edge, torch.tensor((each_weight-np.min(each_weight))/(np.percentile(each_weight, 75)-np.percentile(each_weight, 25))))
		weight[0] = 1
		edge_index_list.append(edge)
		edge_weight_list.append(weight.float())
		if use_depth_divide:
			level_list = e2[(e1==0) | (e1==e2)]
			edge_depth = torch.ones(len(e1))
			source = e1
			DEPTH = 1
			while source.sum().item()>0:
				source = source[torch.isin(source, level_list, invert=True)]
				edge_depth += DEPTH * (torch.isin(e1, level_list)).int()
				level_list = e2[torch.isin(e1, level_list)]
				DEPTH+=1
				
			edge_depth = torch.cat([edge_depth, torch.ones(len(weight)-len(edge_depth))])
			data[b_i].edge_attr = edge_depth.squeeze(0)
		edge_attr_list.append(edge_depth.float().squeeze(0))
	return data, edge_index_list, edge_weight_list, edge_attr_list


def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		