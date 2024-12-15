import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections


data = pd.read_pickle('raw_data/pol_id_twitter_mapping.pkl') #int-string
type_dict = {}
news_id = []
user_id = []

#total 41054 nodes; #news node: 314;  #user node: 40740
for i in data:
    if(data[i].startswith('p')):
        type_dict[i] = 'news'
        news_id.append(i)
    else:
        type_dict[i]='user'
        user_id.append(i)


news_class_dict = {}
nodeId_graphId = np.load('raw_data/node_graph_id.npy')
graphId_graphClass = np.load('raw_data/graph_labels.npy')
fa = 0
tr = 0
fake_news_list = []
true_news_list = []
for i in news_id:
    news_class_dict[i]=nodeId_graphId[i]
    s = ''
    if(int(graphId_graphClass[news_class_dict[i]]) == 1):
        s = 'fake'
        fake_news_list.append(i)
        fa += 1
    else:
        s = 'true'
        true_news_list.append(i)
        tr += 1
    news_class_dict[i] = s
    # label=1: fake(157)   label=0: true(157)

print("fake: ", fake_news_list)
print("true: ", true_news_list)

f = "A.txt"
G = nx.read_edgelist(f, nodetype = int, delimiter = ',', data=False)
# for i in range(41054):
#     if G.has_node(i)==False:
#         G.add_node(i)


nx.set_node_attributes(G,"type",type_dict)
nx.set_node_attributes(G,"class",news_class_dict)
nx.write_gml(G,"politfact.gml")


#############################################################################

fake_news_degree_dict = G.degree(fake_news_list)
true_news_degree_dict = G.degree(true_news_list)
news_degree_dict = G.degree(news_id)

fake_news_degree_dis_dict = {}
true_news_degree_dis_dict = {}
news_degree_dis_dict = {}

########
for i in range(451):
    true_news_degree_dis_dict[i]=0
    news_degree_dis_dict[i]=0
for i in range(283):
    fake_news_degree_dis_dict[i]=0
#########

fake_news_total_degree = 0
true_news_total_degree = 0
news_total_degree = 0

for i in fake_news_degree_dict:
    degree = fake_news_degree_dict[i]
    fake_news_total_degree+=degree
    fake_news_degree_dis_dict[degree] +=1

for i in true_news_degree_dict:
    degree = true_news_degree_dict[i]
    true_news_total_degree += degree
    true_news_degree_dis_dict[degree] +=1

for i in news_degree_dict:
    degree = news_degree_dict[i]
    news_total_degree+= degree
    news_degree_dis_dict[degree] +=1

avg_fake_news_degree = 1.0*fake_news_total_degree/len(fake_news_list)
avg_true_news_degree = 1.0*true_news_total_degree/len(true_news_list)
avg_news_degree = 1.0*news_total_degree/len(news_id)

print("The average degree of fake news node is: ",avg_fake_news_degree)
print("The average degree of true news node is: ",avg_true_news_degree)
print("The average degree of news node is: ",avg_news_degree)

f_l = sorted(fake_news_degree_dis_dict)
t_l = sorted(true_news_degree_dis_dict)
n_l = sorted(news_degree_dis_dict)

fake_news_x = []
fake_news_y = []
for key in f_l:
    v = 1.0 * fake_news_degree_dis_dict[key] / fake_news_total_degree
    #if(v==0):continue
    fake_news_x.append(key)
    fake_news_y.append(v)
#print("fake: ",fake_news_x)

true_news_x = []
true_news_y = []
for key in t_l:
    v = 1.0 * true_news_degree_dis_dict[key] / true_news_total_degree
    #if (v == 0): continue
    true_news_x.append(key)
    true_news_y.append(v)

news_x = []
news_y = []
for key in n_l:
    v = 1.0*news_degree_dis_dict[key]/news_total_degree
    #if(v==0):continue
    news_x.append(key)
    news_y.append(v)

fake_clustering_coefficient = nx.average_clustering(G,nodes=fake_news_list)
true_clustering_coefficient = nx.average_clustering(G,nodes=true_news_list)
news_clustering_coefficient = nx.average_clustering(G,nodes =news_id)
print("The average clustering coefficient of fake news node is: ",fake_clustering_coefficient)
print("The average clustering coefficient of true news node is: ",true_clustering_coefficient)
print("The average clustering coefficient of news node is: ",news_clustering_coefficient)

print("The maximum degree of fake news node is {}".format(fake_news_x[-1]))
print("The minimum degree of fake news node is {}\n".format(fake_news_x[0]))
print("The maximum degree of true news node is {}".format(true_news_x[-1]))
print("The minimum degree of true news node is {}\n".format(true_news_x[0]))
print("The maximum degree of news node is {}".format(news_x[-1]))
print("The minimum degree of news node is {}\n".format(news_x[0]))
######################################################
plt.figure(figsize=(6.4,6.4))
#plt.title("Probability Degree Distribution")

plt.subplot(221)
plt.bar(fake_news_x,fake_news_y,color='red',label='fake')
plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
plt.label("probability distribution of fake news node")
#plt.xticks(range(0,300,25))
plt.legend()



plt.subplot(222)
plt.bar(true_news_x,true_news_y,color ='blue',label='true')
plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
#plt.xticks(range(0,460,25))
plt.label("probability distribution of true news node")
plt.legend()


plt.subplot(223)
plt.bar(news_x,news_y,color='green',label='news')
plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
plt.label("probability distribution of news node")
#plt.xticks(range(0,460,25))
plt.legend()



# fig, ax = plt.subplots()
# ax.bar(fake_news_x,fake_news_y,color='red',label='fake')
# plt.legend()
# plt.title('Degree Distribution of fake news')
# plt.savefig("Degree Distribution of fake news.png")

# fig2, ax2 = plt.subplots()
# ax2.bar(true_news_x,true_news_y,color ='blue',label='true')
# plt.legend()
# plt.title('Degree Distribution of true news')
# plt.savefig("Degree Distribution of true news.png")

# fig3, ax3 = plt.subplots()
# ax3.bar(news_x,news_y,color='green',label='news')
# plt.legend()
# plt.title('Degree Distribution of news')
# plt.savefig("Degree Distribution of news.png")
###########################################################
user_degree_dict = G.degree(user_id)
user_degree_dis_dict = {}
user_total_degree = 0
for i in user_degree_dict:
    degree = user_degree_dict[i]
    user_total_degree+= degree
    if degree in user_degree_dis_dict:
       user_degree_dis_dict[degree] +=1
    else:
       user_degree_dis_dict[degree]=0

avg_user_degree = 1.0*user_total_degree/len(user_id)
print("The average degree of user node is: ",avg_user_degree)
u_l = sorted(user_degree_dis_dict)
user_x = []
user_y = []
for key in u_l:
    v = 1.0*user_degree_dis_dict[key]/user_total_degree
    #if(v==0):continue
    user_x.append(key)
    user_y.append(v)
user_clustering_coefficient = nx.average_clustering(G,nodes =user_id)
print("The average clustering coefficient of user node is: ",user_clustering_coefficient)
print("The maximum degree of user node is {}".format(user_x[-1]))
print("The minimum degree of user node is {}".format(user_x[0]))


plt.subplot(224)
plt.bar(user_x,user_y,color='#C12B1B',label='user')
plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
plt.label("probability distribution of user node")
#plt.xticks(range(0,170,25))
plt.legend()
plt.savefig("Probability Degree Distribution.png")