import numpy as np

node = np.load('node_graph_id.npy')
print(node.shape)

graph = np.load("graph_labels.npy")
print(graph.shape)

print("fake: ",sum(graph==1))
print("true: ",sum(graph==0))

path = 'A.txt'
A = np.loadtxt(path,dtype=np.int64,delimiter=',')
print(A.shape)