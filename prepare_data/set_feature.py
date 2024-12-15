import pandas as pd
import numpy as np
import json
import os
import spacy
from set_graph_labels import del_graph_ids
from set_node_graph_id import del_node_ids
import scipy.sparse as sp
import copy

'''
In this file, we delete the news nodes whose articles are no longer available.
Also, we apply feature engineering using the spacy library.

'''

data = pd.read_pickle('raw_data/pol_id_twitter_mapping.pkl')
#{0: 'politifact4190', 1: '14200465',...}


fake_news_list = [18042, 18151, 18267, 18312, 18507, 18855, 19054, 19158, 19280, 19303, 19501, 19791, 19838, 20209, 20304, 20375, 20378, 20609, 20778, 21093, 21128, 21140, 21244, 21405, 21644, 21949, 21977, 22392, 22414, 22695, 22712, 22747, 23012, 23112, 23277, 23316, 23359, 23430, 23511, 23768, 23960, 24045, 24285, 24295, 24639, 24860, 24922, 25386, 25485, 25513, 25761, 25891, 25920, 26113, 26135, 26169, 26246, 26339, 26352, 26662, 26972, 27051, 27349, 27759, 27786, 27894, 27949, 28197, 28259, 28271, 28479, 28499, 28644, 28765, 28987, 29013, 29246, 29293, 29390, 29698, 29884, 30301, 30675, 31151, 31179, 31201, 31243, 31265, 31564, 31629, 31669, 31745, 31757, 31816, 31996, 32005, 32218, 32332, 32400, 32417, 32444, 32700, 33158, 33179, 33356, 33664, 33809, 33823, 34138, 34370, 34444, 34487, 34527, 34548, 34572, 34664, 35071, 35096, 35336, 35354, 35572, 35974, 36120, 36272, 36281, 36757, 36860, 37036, 37342, 37694, 37701, 37762, 37794, 37964, 38397, 38476, 38611, 38615, 38838, 38861, 39169, 39275, 39370, 39373, 39406, 39439, 39699, 39706, 39760, 39767, 39776, 40263, 40276, 40310, 40595, 40729, 40850]
true_news_list = [0, 497, 518, 660, 675, 705, 735, 821, 831, 880, 1313, 1465, 1888, 1925, 2034, 2346, 2436, 2456, 2462, 2502, 2591, 2597, 3089, 3326, 3404, 3505, 3534, 3649, 4090, 4322, 4404, 4549, 4603, 4921, 5318, 5328, 5345, 5387, 5428, 5614, 5652, 5860, 5932, 6092, 6096, 6121, 6173, 6442, 6704, 6716, 7134, 7318, 7799, 7864, 7870, 7874, 7878, 7884, 7912, 7972, 8133, 8137, 8140, 8160, 8172, 8185, 8192, 8199, 8469, 8486, 8798, 8921, 8966, 8988, 9026, 9074, 9203, 9215, 9222, 9410, 9436, 9475, 9510, 9961, 10004, 10023, 10077, 10117, 10123, 10402, 10418, 10435, 10481, 10484, 10614, 10672, 10686, 10713, 10945, 10960, 11191, 11268, 11284, 11300, 11614, 11709, 11733, 12012, 12182, 12515, 12523, 12563, 12615, 12670, 12675, 12684, 12751, 12797, 12874, 12964, 13056, 13172, 13311, 13452, 13822, 13880, 13906, 14263, 14571, 14574, 14824, 15077, 15303, 15325, 15385, 15562, 15947, 16042, 16067, 16077, 16156, 16252, 16712, 16724, 16780, 16965, 17023, 17042, 17082, 17385, 17396, 17481, 17510, 17585, 17643, 17647, 17918]

content_dict = {}
fault = []
for i in data:
    if(data[i].startswith('p')):
        typee = ''
        if i in fake_news_list:
            typee = 'fake'
        else:
            typee = 'real'
        try:
            with open("fakenewsnet_dataset2/politifact/{}/{}/news content.json".format(typee,data[i]),"r",encoding="UTF-8") as f:
                result = json.load(f)
                content_text = result.get('text')
                content_title = result.get('title')
                content = content_title + content_text
                content_dict[i] = content
        except:
            fault.append(i)
            pass


print(fault)
#print(a,b,c)
# fault = [3505, 5428, 10713, 11709, 16780, 20378, 21949, 21977, 23359, 26135, 26339, 27894, 29698, 31151, 37762, 38611] #16个不行
removed = [0,1888,10123,38615,5345,8988,5932,9436,26169,7878,8798,8133,9510,31745,12523,9961,10481,31816,32444,33179,12684,\
           16724,14571,17042,15562,18267,18855,21140,22712,31243,31669,23277,24860,28259,28644,29013,34138,\
           34370,37036,2456,5860,6716,8140,19303,21093,23768]


for i in removed:
    content_dict.pop(i)

print("number of news: ",len(content_dict))

feature_dict = dict()

# Use spacy to complete feature engineering and get the text embedding.
nlp = spacy.load('en_core_web_lg')
for i in content_dict:
    print(i)
    feature = nlp(content_dict[i])
    feature_dict[i] = np.asarray(feature.vector)


del_graph_id = del_graph_ids()
_,del_node_id = del_node_ids()

data = np.load("raw_data/new_spacy_feature.npz")
features = data['data'].reshape((data['shape']))

for i in feature_dict:
    features[i] = feature_dict[i]

#print(features.shape)

news_list = fake_news_list+true_news_list
news_list2 = copy.deepcopy(news_list)
for i in news_list:
    if i in removed:
        news_list2.remove(i)
print("len(news_list2): ",len(news_list2))


idx = 0
news_id = []
new_features = []
for i in range(len(features)):
    if i in del_node_id:
        continue
    else:
        new_features.append(features[i])
        if i in news_list2:
            news_id.append(idx)
        idx += 1

print(len(news_id))
news_id2 = np.asarray(news_id)
np.save('../data/politifact/news_id.npy', news_id2)

new_features = np.asarray(new_features)
print(new_features.shape)
new_features = sp.csr_matrix(new_features)
sp.save_npz('processed_data/new_spacy_feature.npz',new_features)