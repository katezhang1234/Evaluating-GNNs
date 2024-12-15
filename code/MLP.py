import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F

feature_file_path = "../data/politifact/raw/new_spacy_feature.npz"
node_graph_file_path = '../data/politifact/raw/node_graph_id.npy'
graph_labels_path = "../data/politifact/raw/graph_labels.npy"
node_id_path = "../data/politifact/news_id.npy"

'''
Create Dataset
'''

class MyData(Dataset):

    def __init__(self,feature_file_path,node_graph_file_path,graph_labels_path,node_id_path):
        self.feature_file_path = feature_file_path
        self.node_graph_file_path = node_graph_file_path
        self.graph_labels_path = graph_labels_path
        self.node_id_path = node_id_path

        features = np.load(self.feature_file_path)
        feature = features['data'].reshape((features['shape']))
        node_graph_id = np.load(self.node_graph_file_path)
        graph_labels = np.load(graph_labels_path)
        node_id = np.load(self.node_id_path)
        print(len(node_id))
        self.data = []
        self.label = []
        for id in node_id:
            self.data.append(feature[id])
            graph_id = node_graph_id[id]
            self.label.append(graph_labels[graph_id])


    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]


    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)


'''
Build MLP model
'''
class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()

        self.args = args
        self.linear1 = nn.Linear(args["num_i"], args["num_h"])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(args["num_h"], args["num_h"])  # 2个隐层
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(args["num_h"], args["num_o"])

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

args = {"batch_size": 256,
        "epochs": 30,
        "lr": 0.01,
        "weight_decay": 0.01,
        'device': "cpu",       # cuda:0
        "num_i": 300,
        "num_h": 128,
        "num_o": 2}

dataset = MyData(feature_file_path,node_graph_file_path,graph_labels_path,node_id_path)
print(len(dataset))

TRAIN_SUBSET = 0.7
VAL_SUBSET = 0.1
num_training = int(len(dataset) * TRAIN_SUBSET)
num_val = int(len(dataset) * VAL_SUBSET)
num_test = len(dataset) - (num_training + num_val)
print(num_training)
print(num_val)
print(num_test)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args["batch_size"], shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args["batch_size"], shuffle=False)
test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False)


model = MLP(args)
print(model)
model = model.to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
cost = torch.nn.CrossEntropyLoss()

###################################
'''
Train MLP
'''

train_accuracies = []
for epoch in range(args["epochs"]):
    sum_loss = 0
    train_correct = 0
    TP = 0
    TF = 0
    FP =0
    FN = 0
    for data in train_loader:
        #print(data)
        inputs, labels = data  # inputs 维度：[64,1,28,28]
        #     print(inputs.shape)
        #inputs = torch.flatten(inputs, start_dim=1)  # 展平数据，转化为[64,784]
        #     print(inputs.shape)
        inputs = inputs.to(args["device"])
        inputs = inputs.to(torch.float)
        labels = labels.long()
        labels = labels.to(args["device"])
        outputs = model(inputs)
        #print(outputs)
        optimizer.zero_grad()
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()

        _, id = torch.max(outputs.data, 1)
        sum_loss += loss.data
        train_correct += torch.sum(id == labels)
        TP += torch.sum(labels & id)
        TF += torch.sum(labels & (1 - id))
        FP += torch.sum((1 - labels) & id)
        FN += torch.sum((1 - labels) & (1 - id))

    Precision = 1.0 * TP / (TP + FP)
    Recall = 1.0 * TP / (TP + FN)
    F1 = (2.0 * Precision * Recall) / (Precision + Recall)
    Accuracy = train_correct / num_training
    train_accuracies.append(Accuracy)
    print('[%d,%d] loss:%.03f' % (epoch + 1, args["epochs"], sum_loss / len(train_loader)))
    print(f'       acc: {Accuracy:.4f}, f1: {F1:.4f}, '
          f'precision: {Precision:.4f}, recall: {Recall:.4f}')

print("Train_accuracies = ", train_accuracies)

model.eval()


test_correct = 0
TP = 0
TF = 0
FP = 0
FN = 0
for d in test_loader:
    i, l = d
    print(i.shape)
    print(l.shape)
    i = i.to(args["device"])
    i = i.to(torch.float)
    l = l.long()
    l= l.to(args["device"])

    o = model(i)
    _, id = torch.max(o.data, 1)
    #print(id)
    #print(labels)
    test_correct += torch.sum(id == l)
    TP += torch.sum(l & id)
    TF += torch.sum(l & (1 - id))
    FP += torch.sum((1 - l) & id)
    FN += torch.sum((1 - l) & (1 - id))

Precision = 1.0 * TP / (TP + FP)
Recall = 1.0 * TP / (TP + FN)
F1 = (2.0 * Precision * Recall) / (Precision + Recall)
Accuracy = test_correct / num_test
print(f'       acc: {Accuracy:.4f}, f1: {F1:.4f}, '
      f'precision: {Precision:.4f}, recall: {Recall:.4f}')


