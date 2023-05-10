from trainers.general_trainer import *
from collections import OrderedDict

import json
import sys
import csv
# import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("..")
from data import LoadData
from models.gcn import GCN
from models.hetero_gcn import GCNSep
from torch_geometric.loader import NeighborLoader

def train_gcn_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    print_graphs=False,
    seeds=[1],
    test_dataset=None,
    feed_hidden_layer=False,
    sample_neighbors=False
):
    print(
        "Training gcn dataset={}, test_dataset={}".format(
            dataset, test_dataset
        )
    )
    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    rare_f1_scores = []
    best_epochs = []
    is_rare = "twitch" in dataset.value

    for i in range(len(seeds)):
        set_torch_seed(seeds[i])
        rng = np.random.default_rng(seeds[i])
        print("We run for seed {}".format(seeds[i]))
        data_loader = LoadData(
            dataset,
            rng=rng,
            rng_seed=seeds[i],
            test_dataset=test_dataset,
            split_num_for_dataset = i%10,
            inductive = run_config.inductive
        )
        
        data_loader.train_data = data_loader.train_data.to(device, 'x', 'y')
        num_classes = data_loader.num_classes
        train_features = data_loader.train_features
        train_labels = data_loader.train_labels
        num_nodes = len(train_labels)

        num_train_nodes = (data_loader.train_mask == True).sum().item()        
        batch_size = run_config.batch_size if run_config.batch_size else num_train_nodes
        num_neighbors = [25] * run_config.num_hidden if sample_neighbors else [-1] * run_config.num_hidden

        train_loader = NeighborLoader(
            data_loader.train_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data_loader.train_mask,
            **{'shuffle': True}
        )
        
        data_loader.val_data = data_loader.val_data.to(device, 'x', 'y')
        val_loader = NeighborLoader(
            data_loader.val_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data_loader.val_mask,
            **{'shuffle': True}
        )
        
        if num_classes > 2:
            is_rare = False
        if data_loader.test_dataset:
            test_dataset = data_loader.test_dataset
        
        # Create a model based on the config and input size
        if run_config.hetero:
            model = GCNSep(input_size=train_features.size(1),
                        hidden_size=run_config.hidden_size,
                        output_size=num_classes,
                        dropout=run_config.dropout,
                        num_hidden=run_config.num_hidden,
                        feed_hidden_layer=feed_hidden_layer)
        else:
            model = GCN(input_size=train_features.size(1),
                        hidden_size=run_config.hidden_size,
                        output_size=num_classes,
                        dropout=run_config.dropout,
                        num_hidden=run_config.num_hidden,
                        feed_hidden_layer=feed_hidden_layer)
                        
        if run_config.parse_communication_results:
            utils.parse_communication_gnn(model, 'mmlp_gcn_nl_2', dataset)
            return 

        kwargs = {}
        
        trainer = Trainer(model, rng, seed=seeds[i])
        
        if run_config.calculate_communication:
            comm_stats = {}
            comm_stats_with_server = {}
        
            for x in range(num_nodes):
                layers = np.zeros(len(model.linear_layers_list))
                layers_server = np.zeros(len(model.linear_layers_list))
                embeddings = np.zeros(len(model.linear_layers_list) + 1)
                gradients = np.zeros(len(model.linear_layers_list))
                nei_embeddings = [set() for _ in range(len(model.linear_layers_list) + 1)]
                            
                comm_stats[x] =  {
                    "layers": layers,
                    "embeddings": embeddings,
                    "gradients": gradients,
                    "nei_embeddings": nei_embeddings,
                }
                
                comm_stats_with_server[x] = {
                    "layers": layers_server,
                }                    
        
            kwargs["comm_stats"] = comm_stats
            kwargs["comm_stats_with_server"] = comm_stats_with_server
            kwargs["is_mmlp"] = False
            kwargs["train_model"] = -1
        
        val_loss, val_acc, best_epoch = trainer.train(
            dataset,
            train_loader,
            val_loader,
            device,
            run_config,
            test_dataset=test_dataset,
            sample_neighbors=sample_neighbors,
            feed_hidden_layer=feed_hidden_layer,
            kwargs=kwargs
        )

        data_loader.test_data = data_loader.test_data.to(device, 'x', 'y')
        test_loader = NeighborLoader(
            data_loader.test_data,
            num_neighbors=num_neighbors,
            batch_size=data_loader.test_mask.sum().item(),
            input_nodes=data_loader.test_mask,
            **{'shuffle': True}
        )
        
        if run_config.calculate_communication:
            comm_result = {}
            comm_result_with_server = {}
            
            for node in kwargs["comm_stats"]:
                comm_result[node] = {
                    "layers": kwargs["comm_stats"][node]["layers"].tolist(),
                    "embeddings": kwargs["comm_stats"][node]["embeddings"].tolist(),
                    "gradients": kwargs["comm_stats"][node]["gradients"].tolist()
                }
                
                comm_result_with_server[node] = {
                    "layers": kwargs["comm_stats_with_server"][node]["layers"].tolist()
                }
            
        test_loss, test_acc, f1_score, rare_f1_score, out_labels, logits = trainer.evaluate(
            test_loader, run_config, is_rare=is_rare, kwargs=kwargs
        )  
        
        if run_config.calculate_communication:
            with open(f"comm_results/{dataset}_{model.model_name}.json", "w") as f:
                json.dump(comm_result, f, indent = 4, sort_keys=True)
                
            with open(f"comm_results/{dataset}_{model.model_name}_with_server.json", "w") as f:
                json.dump(comm_result_with_server, f, indent = 4, sort_keys=True)
            
        all_outputs.append((out_labels, logits))
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        f1_scores.append(f1_score)
        best_epochs.append(best_epoch)
        if is_rare:
            rare_f1_scores.append(rare_f1_score)

    train_stats = TrainStats(
        run_config,
        dataset,
        model.model_name,
        all_outputs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores,
        test_dataset,
    )
    train_stats.print_stats()

    return train_stats