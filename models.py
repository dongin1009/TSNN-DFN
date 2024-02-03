import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, global_max_pool, global_mean_pool0
from torch_geometric.data import Batch
import torch_scatter
import math
import copy as cp

from utils import eval_deep

def import_models(args):
	if args.model == 'TSNN':
		model = TSNN(args)
	elif args.model == 'UPFD-gcn':
		model = UPFD(args, concat=args.concat)
	elif args.model == 'UPFD-gat':
		model = UPFD(args, concat=args.concat)
	elif args.model == 'UPFD-sage':
		model = UPFD(args, concat=args.concat)
	elif args.model == 'UPFD-transformer':
		model = UPFD(args, concat=args.concat)
	elif args.model == 'BiGCN':
		model = BiGCN(args.num_features, args.nhid, args.num_classes)
	elif args.model == 'GCNFN':
		model = GCNFN(args, concat=args.concat)
	return model

class TSNN(torch.nn.Module):
	def __init__(self, args):
		super(TSNN, self).__init__()
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.dropout_p = args.dropout_p
		self.pre_padding = False
		self.use_time_decay_score = args.use_time_decay_score
		self.use_depth_divide = args.use_depth_divide
		self.seq_layer_type = args.seq_layer_type

		self.leafconv1 = GATConv(self.num_features, self.num_features)
		self.conv1 = GCNConv(self.num_features, 2*self.nhid)
		self.conv2 = GCNConv(2*self.nhid, self.nhid)
		
		if args.seq_layer_type == 'lstm':
			self.seq_layer = torch.nn.LSTM(self.num_features, self.nhid//2, num_layer=args.num_seq_layers, batch_first=True, bidirectional=True)
		elif args.seq_layer_type == 'gru':
			self.seq_layer = torch.nn.GRU(self.num_features, self.nhid//2, num_layer=args.num_seq_layers, batch_first=True, bidirectional=True)
		elif args.seq_layer_type == 'transformer':
			self.pe = PositionalEncoding(self.num_features, dropout=0.1)
			self.seq_layer = torch.nn.Transformer(d_model=self.num_features, nhead=2, num_encoder_layers=args.num_seq_layers, num_decoder_layers=args.num_seq_layers, dim_feedforward=self.num_features//2, batch_first=True)
			self.pre_padding = True
		elif args.seq_layer_type == 'transformer_encoder':
			self.pe = PositionalEncoding(self.num_features, dropout=0.1)
			self.seq_layer = torch.nn.TransformerEncoderLayer(d_model=self.num_features, nhead=2, batch_first=True)
			self.seq_layer = torch.nn.TransformerEncoder(self.sequential_layer, num_layers=args.num_seq_layers)
		self.lin = torch.nn.Linear(self.num_features, self.nhid)
				
		self.cls = torch.nn.Linear(2*self.nhid, self.num_classes)
		
	def forward(self, data):
		x, edge_index, time_decay_score, depth, batch, num_graphs = data.x, data.edge_index, data.edge_weight, data.edge_attr, data.batch, data.num_graphs
		seq_x = torch_geometric.utils.unbatch(x, batch)

		if self.pre_padding:
			seq_x = tuple(map(lambda s: s.flip(0), seq_x))
			seq_x = pad_sequence(seq_x, batch_first=True)
			seq_x = seq_x.flip(1)
		else:
			seq_x = pad_sequence(seq_x, batch_first=True)
		pad_mask = seq_x.sum(-1)==0
		#edge_attr = None

		news_index = torch.stack([(batch == idx).nonzero().squeeze()[0] for idx in range(num_graphs)])
		leaf_index = (torch.isin(edge_index[0], news_index, invert=True) & torch.isin(edge_index[1], news_index, invert=True))
		leaf_edge_index = edge_index[:, leaf_index]
		
		x = F.relu(self.leafconv1(x, leaf_edge_index))

		if self.use_time_decay_score:
			if self.use_depth_divide:
				x = F.relu(self.conv1(x, edge_index, time_decay_score/depth))
				x = F.dropout(x, p=self.dropout_p, training=self.training)
				x = F.relu(self.conv2(x, edge_index, time_decay_score/depth))
			else:
				x = F.relu(self.conv1(x, edge_index, time_decay_score))
				x = F.dropout(x, p=self.dropout_p, training=self.training)
				x = F.relu(self.conv2(x, edge_index, time_decay_score))
		else:
			x = F.relu(self.conv1(x, edge_index))
			x = F.dropout(x, p=self.dropout_p, training=self.training)
			x = F.relu(self.conv2(x, edge_index))

		if self.seq_layer_type == 'transformer':
			seq_x = self.pe(seq_x*math.sqrt(self.num_features))
			seq_x = self.seq_layer(seq_x, seq_x, src_key_padding_mask=pad_mask, tgt_key_padding_mask=pad_mask)
		elif self.seq_layer_type == 'transformer_encoder':
			seq_x = self.pe(seq_x*math.sqrt(self.num_features))
			seq_x = self.seq_layer(seq_x, src_key_padding_mask=pad_mask)
		elif self.seq_layer_type == 'lstm' or self.sequential_layer == 'gru':
			seq_x, _ = self.sequential_layer(seq_x)
		
		if self.pre_padding:
			seq_x = seq_x[:, -1, :]
		else:
			seq_x = seq_x[:, 0, :]
		S = F.relu(F.dropout(self.lin(seq_x), p=self.dropout_p, training=self.training))

		T = x[news_index] # supernode

		S = S.squeeze()
		out = torch.cat([T, S], dim=1)
		out = F.log_softmax(self.cls(out), dim=-1)
		return out

class PositionalEncoding(torch.nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = torch.nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		Args:
			input: Tensor, shape [seq_len, batch_size, embedding_dim] 
		"""
		x = x.permute(1, 0, 2)
		x = x + self.pe[:x.size(0)]
		x = self.dropout(x)
		x = x.permute(1, 0, 2)
		return x

# def reset_weights(m):
# 	if isinstance(m, GCNConv) or isinstance(m, GATConv) or isinstance(m, torch.nn.GRU) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
# 		m.reset_parameters()

@torch.no_grad()
def compute_test(loader, model, is_listed=False, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if is_listed:
			data = Batch.from_data_list(data)
		data = data.to(model.device)
		out = model(data)
		y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test



#### baseline models ####
class UPFD(torch.nn.Module):
	def __init__(self, args, concat=False):
		super(UPFD, self).__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.model = args.model
		self.concat = concat

		if args.model == 'UPFD-gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif args.model == 'UPFD-sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif args.model == 'UPFD-gat':
			self.conv1 = GATConv(self.num_features, self.nhid)
		elif args.model == 'UPFD-transformer': ## ADD 'TransformerConv' function from PyG.
			self.conv1 = TransformerConv(self.num_features, self.nhid)

		if self.concat:
			self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
			self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)
		
	def forward(self, data):

		x, edge_index, batch = data.x, data.edge_index, data.batch

		edge_attr = None

		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = global_max_pool(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))

		x = F.log_softmax(self.lin2(x), dim=-1)

		return x

class RumorGCN(torch.nn.Module):
	"""
	The Bi-GCN is adopted from the original implementation from the paper authors 

	Paper: Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
	Link: https://arxiv.org/pdf/2001.06362.pdf
	Source Code: https://github.com/TianBian95/BiGCN
	-- Implemented Code: https://github.com/safe-graph/GNN-FakeNews/blob/main/gnn_model/bigcn.py
	"""
	def __init__(self, in_feats, hid_feats, out_feats, types='TD'):
		super(RumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
		self.types = types ## ADD types of 'TopDown' or 'BottomUp' for combining to a single function.
	
	def forward(self, data):
		if self.types == 'TD':
			x, edge_index = data.x, data.edge_index
		elif self.types == 'BU':
			x, edge_index = data.x, data.BU_edge_index
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)
		rootindex = data.root_index
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
		batch_size = max(data.batch) + 1

		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)
		x = torch_scatter.scatter_mean(x, data.batch, dim=0)

		return x

class BiGCN(torch.nn.Module):
	def __init__(self, args):
		super(BiGCN, self).__init__()
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes

		self.TDrumorGCN = RumorGCN(self.num_features, self.nhid, self.nhid, types='TD')
		self.BUrumorGCN = RumorGCN(self.num_features, self.nhid, self.nhid, types='BU')
		self.fc = torch.nn.Linear((self.nhid+self.nhid) * 2, self.num_classes)

	def forward(self, data):
		TD_x = self.TDrumorGCN(data)
		BU_x = self.BUrumorGCN(data)
		x = torch.cat((TD_x, BU_x), 1)
		x = self.fc(x)
		x = F.log_softmax(x, dim=1)
		return x


class GCNFN(torch.nn.Module):
	"""

	GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder; 
	the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional 
	comment word2vec (spaCy) embeddings plus 10-dimensional profile features 

	Paper: Fake News Detection on Social Media using Geometric Deep Learning
	Link: https://arxiv.org/pdf/1902.06673.pdf


	Model Configurations:

	Vanilla GCNFN: args.concat = False, args.feature = content
	UPFD-GCNFN: args.concat = True, args.feature = spacy
	
	--Implemented Code: https://github.com/safe-graph/GNN-FakeNews/blob/main/gnn_model/gcnfn.py
	"""
	def __init__(self, args, concat=False):
		super(GCNFN, self).__init__()
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.concat = concat

		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

		self.fc1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		if self.concat:
			self.fc0 = torch.nn.Linear(self.num_features, self.nhid)
			self.fc1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		self.fc2 = torch.nn.Linear(self.nhid, self.num_classes)


	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.fc0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.fc1(x))

		x = F.log_softmax(self.fc2(x), dim=-1)

		return x