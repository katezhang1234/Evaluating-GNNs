
import argparse
import time
from tqdm import tqdm
import copy as cp
import warnings
warnings.filterwarnings("ignore")

# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader


from data_loader import *
from eval_helper import *

'''
Note: This code refers to UPFD code, the link is https://github.com/safe-graph/GNN-FakeNews
'''


"""

Implementation for GNN variants:

GCN, GAT, GATv2, GraphSAGE, and GIN

The GCN, GAT, and GraphSAGE implementation

"""


class Model(torch.nn.Module):
	def __init__(self, args, concat=False):
		super(Model, self).__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.dropout_ratio = args.dropout_ratio
		self.model = args.model
		self.concat = concat
		self.heads = 1

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)
		elif self.model == "gatv2":
			self.conv1 = GATv2Conv(self.num_features, self.nhid, heads = self.heads)

		if self.concat:
			self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
			self.lin1 = torch.nn.Linear(self.nhid * (self.heads + 1), self.nhid)

		self.lin2 = torch.nn.Linear(self.nhid * self.heads, self.num_classes)

	def forward(self, data):

		x, edge_index, batch = data.x, data.edge_index, data.batch

		edge_attr = None

		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))

		x = F.log_softmax(self.lin2(x), dim=-1)

		return x


@torch.no_grad()
def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not args.multi_gpu:
			data = data.to(args.device)
		out = model(data)
		if args.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test


'''
Set arguments
'''
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')  # cuda: 0, mps, cpu

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=False, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='spacy', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='gcn', help='model type, [gcn, gat, sage, gatv2, gin]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='../data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)


TRAIN_SUBSET = 0.7
VAL_SUBSET = 0.1
print("DATASET SIZE: ", len(dataset))
print(f"TRAINING on {TRAIN_SUBSET * 100}% of data")
print(f"TESTING on {VAL_SUBSET * 100}% of data")

num_training = int(len(dataset) * TRAIN_SUBSET)
num_val = int(len(dataset) * VAL_SUBSET)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)
print("\nLoaded all data\n")

model = Model(args, concat=args.concat)
print(model)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
	# Model training

	min_loss = 1e10
	val_loss_values = []
	best_epoch = 0

	t = time.time()

	''' 
	Train the model
	'''
	model.train()
	for epoch in tqdm(range(args.epochs)):
		print("Epoch = ", epoch)
		loss_train = 0.0
		out_log = []
		for i, data in enumerate(train_loader):
			optimizer.zero_grad()
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)##########
			if args.multi_gpu:
				y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
			else:
				y = data.y
			loss = F.nll_loss(out, y)
			loss.backward()
			optimizer.step()
			loss_train += loss.item()
			out_log.append([F.softmax(out, dim=1), y])
		acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
		[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
		print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
			  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
			  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
			  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

	[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
	print(f'{args.dataset} {args.model} Testing Results:\n'
		  f'acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
		  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')


# Make sure to add: PYTORCH_ENABLE_MPS_FALLBACK=1


# 70% training, 70 epochs. 
# NO concat
# acc: 0.7059, f1_macro: 0.7059, f1_micro: 0.7059, precision: 0.6000, recall: 0.8571, auc: 0.8635, ap: 0.8292

# 20% training, 55 epochs. 
# NO concat
# acc: 0.6497, f1_macro: 0.6366, f1_micro: 0.6497, precision: 0.5693, recall: 0.9255, auc: 0.7665, ap: 0.6974

# 70% training, 30 epochs
# CONCAT
# acc: 0.9020, f1_macro: 0.9014, f1_micro: 0.9020, precision: 0.8077, recall: 1.0000, auc: 0.9714, ap: 0.9419

# 20% training, 30 epochs
# CONCAT
# acc: 0.7345, f1_macro: 0.7311, f1_micro: 0.7345, precision: 0.6440, recall: 0.9255, auc: 0.9048, ap: 0.8878