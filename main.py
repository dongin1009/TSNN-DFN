import argparse
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch
from sklearn.model_selection import KFold, train_test_split
import os
import sys
sys.path.append(os.getcwd())
import logging

from models import import_models, compute_test
from utils.data_loader import *
from utils.eval_helper import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_p', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--concat', type=lambda s: s.lower() in ['true'], default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--feature', type=str, default='spacy', help='feature type, ["profile", "spacy", "bert", "content"]')
parser.add_argument('--model', type=str, default='TSNN', help='model type, ["TSNN", "UPFD-gcn", "UPFD-gat", "UPFD-sage", "UPFD-transformer", "BiGCN", "GCNFN"]')
parser.add_argument('--es_patience', type=int, default=10)
parser.add_argument('--use_time_decay_score', action='store_true')
parser.add_argument('--use_depth_divide', action='store_true')
parser.add_argument('--seq_layer_type', type=str, default='transformer', help='sequential_layer type, ["transformer", "transformer_encoder", "lstm", "gru"]')
parser.add_argument('--num_seq_layers', type=int, default=2)
args = parser.parse_args()

dataset = FNNDataset(root='data', name=args.dataset, feature=args.feature if len(args.feature.split(', '))==1 else args.feature.split(', '), empty=False, transform=ToUndirected())

if args.model=='TSNN':
	time_dict = make_temporal_weight(dataset, name=args.dataset)
	dataset, edge_index_list, edge_weight_list, edge_attr_list = make_edge_weight(dataset, time_dict, unit='minute', use_depth=args.use_depth_divide)
	dataset_list = []
	for i in range(len(dataset)):
		each_dataset = dataset[i]
		each_dataset.edge_index = edge_index_list[i]
		each_dataset.edge_weight = edge_weight_list[i]
		each_dataset.edge_attr = edge_attr_list[i]
		dataset_list.append(each_dataset)
	loader = DataListLoader
	args.is_listed = True
else:
	loader = DataLoader
	args.is_listed = False

training_set, test_set, train_y, _ = train_test_split(dataset_list, dataset.data.y, test_size=0.25, random_state=42, stratify=dataset.data.y)
#training_set, validation_set, test_set = dataset[dataset.train_idx], dataset[dataset.val_idx], dataset[dataset.test_idx]

###
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
#print(args)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'), datefmt='%Y-%m-%d %H:%M:%S')
logger.info("Argument configurations:")
for arg in vars(args):
	logger.info(f"  {arg}: {getattr(args, arg)}")

if __name__ == '__main__':
	kfold = KFold(n_splits=5, shuffle=True, random_state=42)
	t0 = time.time()
	set_seed(args.seed) #777
	fold_list = ['1st', '2nd', '3rd', '4th', '5th']
	avg_acc, avg_f1_macro, avg_f1_micro, avg_precision, avg_recall, avg_auc, avg_ap = [], [], [], [], [], [], []
	for fold, (train_idx, valid_idx) in zip(fold_list, kfold.split(training_set, train_y)):
		print(f" [{fold} fold training start!]")
		t1 = time.time()
		train_loader = loader([training_set[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
		val_loader = loader([training_set[i] for i in valid_idx], batch_size=args.batch_size, shuffle=False)
		test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

		model = import_models(args)
		for layer in model.children():
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

		model = model.to(args.device)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		early_stopping = EarlyStopping(patience=args.es_patience, verbose=True, path=f'cv_{fold}_{args.dataset[:3]}_{args.model}.pt')

		logger.info(f"  Model Parameters': {sum(p.numel() for p in model.parameters())}")

		for epoch in tqdm(range(1, args.epochs+1)):
			model.train()
			loss_train = 0.0
			out_log = []
			for i, data in enumerate(train_loader):
				optimizer.zero_grad()
				if args.is_listed:
					data = Batch.from_data_list(data)
				data = data.to(args.device)
				out = model(data)
				y = data.y
				loss = F.nll_loss(out, y)
				loss.backward()
				optimizer.step()
				loss_train += loss.item()
				out_log.append([F.softmax(out, dim=1), y])
			acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)

			[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader, model, is_listed=args.is_listed)
			print(f'  loss_train: {loss_train:.2f}, acc_train: {acc_train:.2f},'
				f' recall_train: {recall_train:.2f}, auc_train: {auc_train:.2f},'
				f' loss_val: {loss_val:.2f}, acc_val: {acc_val:.2f},'
				f' recall_val: {recall_val:.2f}, auc_val: {auc_val:.2f}')

			early_stopping(loss_val, model)
			print("")
			if early_stopping.early_stop:
				print(f" Early stopping at {epoch-early_stopping.patience} !")
				break
		
		model.load_state_dict(torch.load(f'cv_{fold}_{args.dataset[:3]}_{args.model}.pt' if args.es_patience>0 else None))
		test_results, test_loss = compute_test(test_loader, model)
		[acc, f1_macro, f1_micro, precision, recall, auc, ap] = test_results
		avg_acc.append(f"{acc:.2f}")
		avg_f1_macro.append(f"{f1_macro:.2f}")
		avg_f1_micro.append(f"{f1_micro:.2f}")
		avg_precision.append(f"{precision:.2f}")
		avg_recall.append(f"{recall:.2f}")
		avg_auc.append(f"{auc:.2f}")
		avg_ap.append(f"{ap:.2f}")
		print(f" [{fold} fold test results]")
		print(f"  Elapsed time : {time.time() - t1} sec")
		print(f"  Test set results on {fold} fold: acc: {acc:.2f}, f1_macro: {f1_macro:.2f}, f1_micro: {f1_micro:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, auc: {auc:.2f}, ap: {ap:.2f}")		
	print("")
	print("[Summary on 5-fold CV] : ")
	print(f"  [Time] Total elapsed time : {time.time() - t0} sec, Avg : {(time.time() - t0)/5} sec")
	print(f"  [Each] Test set results on each fold: acc: {avg_acc}, f1_macro: {avg_f1_macro}, f1_micro: {avg_f1_micro}, precision: {avg_precision}, recall: {avg_recall}, auc: {avg_auc}, ap: {avg_ap}")
	print(f"  [Avg]  Test set results averaging on 5-fold CV: acc: {avg_acc.mean():.2f}, f1_macro: {avg_f1_macro.mean():.2f}, f1_micro: {avg_f1_micro.mean():.2f}, precision: {avg_precision.mean():.2f}, recall: {avg_recall.mean():.2f}, auc: {avg_auc.mean():.2f}, ap: {avg_ap.mean():.2f}")