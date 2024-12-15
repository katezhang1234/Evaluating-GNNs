from set_node_graph_id import del_node_ids,re_index
import numpy as np

path = 'raw_data/A.txt'
new_path = 'processed_data/A.txt'
A = np.loadtxt(path,dtype=np.int64,delimiter=',')

_,del_node_id = del_node_ids()
new_index = re_index()

new_A = []
for i in A:
    if i[0] in del_node_id:
        continue
    new_A.append([new_index[i[0]],new_index[i[1]]])

# for i in A:
#     if i[0] in del_node_id:
#         continue
#     new_A.append(i)

print(len(new_A))
np.savetxt(new_path,new_A,fmt='%i',delimiter=', ')
