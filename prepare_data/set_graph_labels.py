import numpy as np

# Gets list of graph_ids to delete from the list of news_ids
def del_graph_ids():
    # del news id
    del_list = [0, 1888, 10123, 38615, 5345, 8988, 5932, 9436, 26169, 7878, 8798, 8133, 9510, 31745, 12523, 9961, 10481,31816, 32444, 33179, 12684, \
                16724, 14571, 17042, 15562, 18267, 18855, 21140, 22712, 31243, 31669, 23277, 24860, 28259, 28644, 29013,34138, \
                34370, 37036, 2456, 5860, 6716, 8140, 19303, 21093, 23768, 3505, 5428, 10713, 11709, 16780, 20378,21949, \
                21977, 23359, 26135, 26339, 27894, 29698, 31151, 37762, 38611]
    del_graph_id = []

    nodeId_graphId = np.load('raw_data/node_graph_id.npy')
    for i in del_list:
        graph_id = nodeId_graphId[i]
        del_graph_id.append(graph_id)
    return del_graph_id

# graph_id = del_graph_ids()
# print(graph_id)
#[0, 12, 88, 294, 36, 73, 42, 80, 212, 56, 70, 60, 82, 248, 110, 83, 92, 250, 257, 260, 115, 143, 128, 147, 135, 159, 162, 178, 187, 243, 247, 191, 202, 225, 229, 232, 265, 266, 284, 17, 41, 49, 62, 166, 176, 196, 25, 38, 97, 105, 144, 173, 182, 183, 193, 211, 214, 222, 236, 240, 288, 293]

del_graph_id = del_graph_ids()

#graph_label
graph_labels = np.load('raw_data/graph_labels.npy')
new_graph_labels = np.delete(graph_labels , del_graph_id)
np.save("processed_data/graph_labels.npy", new_graph_labels)
# fake: 123     true: 129