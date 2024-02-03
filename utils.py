import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score

class EarlyStopping:
	def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint.pt'):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = float("inf")
		self.delta = delta
		self.path = path

	def __call__(self, val_loss, model):
		if self.patience > 0:
			score = -val_loss

			if self.best_score is None:
				self.best_score = score
				self.save_checkpoint(val_loss, model)
			elif score < self.best_score + self.delta:
				self.counter += 1
				print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
				if self.counter >= self.patience:
					self.early_stop = True
			else:
				self.best_score = score
				self.save_checkpoint(val_loss, model)
				self.counter = 0

	def save_checkpoint(self, val_loss, model):
		if self.verbose:
			print("")
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss


def eval_deep(log, loader):
	"""
	Evaluating the classification performance given mini-batch data
	"""

	# get the empirical batch_size for each mini-batch
#	data_size = len(loader.dataset.indices)
	data_size = len(loader.dataset)
	batch_size = loader.batch_size
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]

	assert len(log) == len(size_list)

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1_macro += f1_score(y, pred_y, average='macro') * size
		f1_micro += f1_score(y, pred_y, average='micro') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap