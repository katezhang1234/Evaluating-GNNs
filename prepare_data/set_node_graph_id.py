from set_graph_labels import del_graph_ids
import numpy as np

def del_node_ids():
    del_graph_id = del_graph_ids()
    node_graph_id = np.load('raw_data/node_graph_id.npy')
    new_node_graph_id = []
    del_node_id = []

    for i in range(len(node_graph_id)):
        if node_graph_id[i] in del_graph_id:
            del_node_id.append(i)
            continue
        else:
            new_node_graph_id.append(node_graph_id[i])

    return new_node_graph_id, del_node_id

def re_index():
    new_node_graph_id, del_node_id = del_node_ids()
    new_index = dict()
    node_graph_id = np.load('raw_data/node_graph_id.npy')
    index = 0
    for i in range(len(node_graph_id)):
        if i in del_node_id:
            continue
        new_index[i] = index
        index += 1
    return new_index



new_node_graph_id, del_node_id = del_node_ids()

rank = -1
pre = 0
new_node_graph_id2 = []
for i in new_node_graph_id:
    if i != pre:
        rank += 1
        pre = i
    new_node_graph_id2.append(rank)

new_node_graph_id2 = np.asarray(new_node_graph_id2)
del_node_id = np.asarray(del_node_id)
np.save("processed_data/node_graph_id.npy", new_node_graph_id2)
np.save("processed_data/del_node_id.npy", del_node_id)