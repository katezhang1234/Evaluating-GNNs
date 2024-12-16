# Detecting Fake News Using Graph Neural Networks

## Models
* All code for GNN models is in code/gnn.py, which refers to the UFPD code at https://github.com/safe-graph/GNN-FakeNews.
* The file contains the code for 6 models: MLP, GCN, GAT, GATv2, GraphSAGE, and GIN.  
* To train and test different models, you can specify the argument --model as 'gcn', 'gat', 'gatv2', 'sage', or 'gin' when running code/gnn.py.
* Other parameters such as learning rate, epochs, dropout rate, etc. can be specified with additional arguments. 

## Data
* The raw data provided by UPFD has been listed in preprare_data/raw_data, and since some articles are no longer available, we need to process the data, the code list in the prepare_data folder, and prefix with "set".
* The processed data is listed in preprare_data/processed_data and data/politifact/raw, the data in these two folders are the same.
* We also need to process the data to make it suitable for the input of GNN, and the processed data is in data/politifact/processed. 

## Network Analysis
* The code for building network is M1.py
* politifact.gml is the graph we built.
