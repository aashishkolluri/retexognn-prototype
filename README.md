# Retexo

A prototype implementation for RetexoGNNs---the first neural networks for communication-efficient training on fully-distributed graphs.

We suggest using Anaconda for managing the environment
## Setting up conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
conda activate [env_name]
```

## Quick Start: Training a single model

### Details of the datasets and architectures before training
We have considered the following datasets and architectures

#### Datasets
&nbsp; Homophilous: cora, citeseer, pubmed, facebook_page, lastfm_asia \
&nbsp; Heterophilous: wiki-cooc, roman-empire (We borrow these datasets from an anonymous ICLR [submission](https://openreview.net/forum?id=tJbbQfw-5wv))

#### Architectures
&nbsp;   MLP\
&nbsp;   GNNs: GCN, GraphSAGE, GAT\
&nbsp;   RetexoGNNs: RetexoGCN, RetexoSAGE, RetexoGAT

### Run training for a single model 

#### Train a GNN on homophilous dataset 

`python src/main.py --dataset [Dataset] --arch [Architecture] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout] --num_epochs [Num_epochs]`

Here is an example to train a GCN on a homophilous dataset 

`python src/main.py --dataset cora --arch gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.0 --num_epochs 400`

#### Train a RetexoGNN on homophilous dataset 

`python src/main.py --dataset [Dataset] --arch [Architecture] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --nl [#Stacked Pooling model] train --lr [Lr] --dropout [Dropout] --num_epochs [Num_epochs]`

Use mmlp_[gnn] for the architecture option, and nl is the number of models stacked after the first mlp. So, for 3 models stacked together in retexo nl will be 2. Here is an example to train a RetexoGCN (mmlp_gcn) on a homophilous dataset

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 train --lr 0.01 --dropout --num_epochs 400`

#### Heterophilous and Inductive
In order to train a model on heterphilous dataset, add `--hetero` in the same command used for homophilous dataset. Here is an example to train a GCN on heterophilous dataset

`python src/main.py --dataset wiki-cooc --hetero --arch gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.0 --num_epochs 400`

Similarly, you can train a Retexo model on a heterophilous dataset.

By Default the training will proceed in transductive setting. To train a model in the inductive setting, add the `--inductive` option.

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --inductive --hidden_size 256 --num_hidden 2 --nl 2 --sample_neighbors --batch_size 512 --early_stopping train --lr 0.01 --dropout 0.0 --num_epochs 400`

#### Miscellaneous

You can also run for multiple seeds using the `--num_seeds` option. In order to do neighbor sampling, add the `--sample_neighbors`. You can also fix the batch size by adding `--batch_size [Batch_Size]`.

`python src/main.py --dataset cora --arch mmlp_gcn --num_seeds 10 --hidden_size 256 --num_hidden 2 --nl 2 --sample_neighbors --batch_size 512 train --lr 0.01 --dropout 0.0 --num_epochs 400`

Add the `--early_stopping` option for stopping early based on validation accuracy.

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 --sample_neighbors --batch_size 512 --early_stopping train --lr 0.01 --dropout 0.0 --num_epochs 400`

The results are stored in the folder defined in globals.py or the directory specified using the `--outdir` option, by default the result are stored in `./results`. The trained models are stored in the args.outdir/models directory.

## Communication Cost
### Compare communication cost of GNNs and Retexo
You can calculate communication cost (data transferred) by all the nodes in a dataset. You can compare a gnn with 2 hidden layers i.e. `--num_hidden 2` and a retexo with 2 pooling models i.e. `--nl 2`

In order to calculate communication, follow the commands in given sequence (example for comparing communication cost of gcn and mmlp_gcn)
```
mkdir comm_results
python src/main.py --dataset cora --arch gcn --sample_seed 10  --hidden_size 128 --num_hidden 2 --calculate_communication --batch_size 512 --sample_neighbors train --lr 0.075 --dropout 0.0 --num_epochs 400
python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 --calculate_communication --batch_size 512 --sample_neighbors train --lr 0.005 --dropout 0.0 --num_epochs 400

python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 --parse_communication_results --batch_size 512 --sample_neighbors train --lr 0.005 --dropout 0.0 --num_epochs 400
python src/main.py --dataset cora --arch gcn --sample_seed 10 --hidden_size 128 --num_hidden 2 --parse_communication_results --batch_size 512 --sample_neighbors train --lr 0.075 --dropout 0.0 --num_epochs 400
```

This will create two following files in `comm_results` --- `plot_values_{GNN_Model}_{Dataset}.csv` and `plot_values_{Retexo_Model}_{Dataset}.csv`. These files will be `plot_values_gcn_cora.csv` and `plot_values_mmlp_gcn_cora.csv` respectively. Both the files contains the data transferred (in MB) by every node in the dataset for GNN and Retexo respectively in the format `node_id, data_transferred` for every node. It will also generate a plot from these values in `comm_results` only named `comm_size_{GNN_Model}_{Dataset}.png` showcasing a pictorial representation of data transferred by node for GNNs and Retexo. In this case of example, it will be `comm_size_gcn_cora.png`.

Moreover, you can compare communication of given architectures on their best hyperparameters. Make sure to keep the `--num_hidden` for GNN and `--nl` for Retexo to 2.


