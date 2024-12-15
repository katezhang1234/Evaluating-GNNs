# Detecting Fake News Using Deep Neural Networks

## Models
* To check the performance of models, please run code/gnn.py
* The file contains the code for four models: GCN, GAT, GraphSAGE, and MLP. 
* The code for GNNs is in code/gnn.py, this code refers to UPFD code, the link is https://github.com/safe-graph/GNN-FakeNews.
* To train and test different models, you can specify the argument --model as 'gcn', 'gat', or 'sage' when run the code/gnn.py.
* There are also some other arguments that can be changed to specify different hyperparameters for training and the meaning of these arguments has been elaborated in the code.

## Data
* The raw data provided by UPFD has been listed in preprare_data/raw_data, and since some articles are no longer available, so we need to process the data, the code list in the prepare_data folder, and prefix with "set".
* The processed data is listed in preprare_data/processed_data and data/politifact/raw, the data in these two folders are the same.
* We also need to process the data to make it suitable for the input of GNN, and the processed data is in data/politifact/processed

## Network Analysis
* The code for building network is M1.py
* politifact.gml is the graph we built.
