# Retexo

We suggest using Anaconda for managing the environment
## Setting up conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
conda activate [env_name]
```

## Quick Start: Training a single model

### Run training for a single model 
We have considered the following datasets and architecture \
Datasets: \
&nbsp; Homophilous: cora, citeseer, pubmed, facebook_page, lastfm_asia \
&nbsp; Heterophilous: wiki-cooc, roman-empire (We borrow these datasets from an anonymous ICLR [submission](https://openreview.net/forum?id=tJbbQfw-5wv))

Architectures: \
&nbsp;   MLP :mlp\
&nbsp;   GNNs: gcn, graphSAGE, gat\
&nbsp;   RetexoGNNs: mmlp_gcn, mmlp_sage, mmlp_gat

To Train a gnn model on homophilous dataset 

`python src/main.py --dataset [Dataset] --arch [Architecture] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout] --num_epochs [Num_epochs]`

Here is an example to train a GCN on a homophilous dataset 

`python src/main.py --dataset cora --arch gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.0 --num_epochs 400`

To Train a Retexo model on homophilous dataset 

`python src/main.py --dataset [Dataset] --arch [Architecture] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --nl [#Stacked Pooling model] train --lr [Lr] --dropout [Dropout] --num_epochs [Num_epochs]`

Here is an example to train a MMLP_GCN on a homophilous dataset

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 train --lr 0.01 --dropout --num_epochs 400`

For Retexo models nl is the number of model stacked after the first mlp. So, for 3 models stacked together in retexo nl will be 2.

In order to train a model on heterphious dataset, you need to add `--hetero` in the same command used for homophilous dataset. Here is an example to train a GCN on heterophilous dataset

`python src/main.py --dataset wiki-cooc --hetero --arch gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.0 --num_epochs 400`

Similarly, you can train a Retexo model on a heterophilous dataset.

You can also run for multiple seeds using the `--num_seeds` option. The results are stored in the folder defined in globals.py or the directory specified using the `--outdir` option, by default the result are stored in `./results`. The trained models are stored in the args.outdir/models directory.

In order to neighbor sampling, you have to add `--sample_neighbors` to the train command for all datasets and models. You can also fix the batch size for a every round of training by adding `--batch_size [Batch_Size]` to the train command.

An Example for that would be

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 --sample_neighbors --batch_size 512 train --lr 0.01 --dropout 0.0 --num_epochs 400`

Similarly, you can train all other models for all datasets with sampling of neighbors for a particular batch size. We have also implemented early stopping for the train, you can add `--early_stopping` to train command. An Example for that would be

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --hidden_size 256 --num_hidden 2 --nl 2 --sample_neighbors --batch_size 512 --early_stopping train --lr 0.01 --dropout 0.0 --num_epochs 400`

By Default all training will proceed in transductive setting. To train a model in inductive settings, you have to add `--inductive` to the train command. It would look like

`python src/main.py --dataset cora --arch mmlp_gcn --sample_seed 10 --inductive --hidden_size 256 --num_hidden 2 --nl 2 --sample_neighbors --batch_size 512 --early_stopping train --lr 0.01 --dropout 0.0 --num_epochs 400`

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


