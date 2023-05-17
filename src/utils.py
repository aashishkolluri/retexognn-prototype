from enum import Enum
from collections import OrderedDict
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
import json 
import csv


class Architecture(Enum):
    MLP = "mlp"
    MMLP = "mmlp"
    SimpleMMLP = "simple_mmlp"
    TwoLayerGCN = "2layergcn"
    GCN = "gcn"
    MMLPGCN = "mmlp_gcn"
    MMLPSAGE = "mmlp_sage"
    GraphSAGE = "graphSAGE"
    GAT = "gat"
    MMLPGAT = "mmlp_gat"

    def __str__(self):
        return self.value


class Dataset(Enum):
    Cora = "cora"
    CiteSeer = "citeseer"
    PubMed = "pubmed"
    facebook_page = "facebook_page"
    LastFM = "lastfm_asia"
    WikiCooc = "wiki-cooc"
    Roman = "roman-empire"
    KarateClub = 'karateclub'

    def __str__(self):
        return self.value


def get_seeds(num_seeds, sample_seed=None):
    if num_seeds > 1:
        np.random.seed(1)
        # The range from which the seeds are generated is fixed
        seeds = np.random.randint(0, 1000, size=num_seeds)
        print("We run for these seeds {}".format(seeds))
    else:
        seeds = [sample_seed]
    return seeds

def get_folder_name(run_config, dataset, model_name, seed, epoch=-1, test_dataset=None, feed_hidden_layer=False, sample_neighbors=False, inductive=False):
    if epoch > 0:
        if not test_dataset:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}"
            )
        else:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}_{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}"
            )
    else:
        if not test_dataset:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}"
            )
        else:
            dir_name = (
                f"arch_{model_name}-dataset_{dataset.name}_{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-"
                f"hidden_size_{run_config.hidden_size}-"
                f"num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}"
            )
            
    if feed_hidden_layer:
        dir_name += "-feed_hidden_layer"
    
    if sample_neighbors:
        dir_name += "-sample_neighbors"
    
    if inductive:
        dir_name += "-inductive"
        
    return dir_name


def save_results_pkl(results_dict, outdir, arch, dataset, run_config, feed_hidden_layer=False, sample_neighbors=False, inductive=False):
    result_file = f"{arch}-{dataset}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-num_hidden_{run_config.num_hidden}-dropout_{run_config.dropout}-training"

    if feed_hidden_layer:
        result_file += "-feed_hidden_layer"
        
    if sample_neighbors:
        result_file += "-sampling_neighbors"

    if inductive:
        result_file += "-inductive"

    result_file += ".pkl"
        
    filename = os.path.join(outdir, result_file)

    print("Saved results at {}".format(filename))
    with open(filename, "wb") as f:
        pkl.dump(results_dict, f)

def get_comms_pkl_file_name(model_name, dataset, seed, test_dataset, run_config):
    dir_name =  get_folder_name(
                    run_config,
                    dataset,
                    model_name,
                    seed,
                    test_dataset=test_dataset,
                )
    comms_file = os.path.join(run_config.output_dir, dir_name) + '_comms.txt'
    return comms_file

def save_comms_pkl(comms_file, comms):
    print("Saved comms at {}".format(comms_file))
    with open(comms_file, "wb") as f:
        pkl.dump(comms, f)

def load_comms_pkl(comms_file):
    print("Loading comms from {}".format(comms_file))
    with open(comms_file, "rb") as f:
        comms = pkl.load(f)
    return comms

def construct_model_paths(arch, dataset, run_config, seed, test_dataset):
    hidden_size = run_config.hidden_size
    num_hidden = run_config.num_hidden
    dropout = run_config.dropout
    lr = run_config.learning_rate
    nl = run_config.nl
    model_paths = []
    if (
        arch == Architecture.MMLP
        or arch == Architecture.SimpleMMLP
    ):
        if not test_dataset:
            it = 0
            model_path = (
                f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}"
            )


            model_paths.append(
                os.path.join(model_path, model_path + ".pth")
            )
            for it in range(1, nl + 1):
                model_path = (
                    f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
                    f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}"
                )

                model_paths.append(
                    os.path.join(model_path, model_path + ".pth")
                )
        else:
            it = 0
            model_path = (
                f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}"
            )


            model_paths.append(
                os.path.join(model_path, model_path + ".pth")
            )
            for it in range(1, nl + 1):
                model_path = (
                    f"arch_{arch}_{nl}_{it}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                    f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}"
                )

                model_paths.append(
                    os.path.join(model_path, model_path + ".pth")
                )
    else:
        nl = -1
        if num_hidden == 2:
            if arch == Architecture.GCN:
                arch_name = "2layergcn"
            else:
                arch_name = "mlp"
        elif num_hidden == 3:
            if arch == Architecture.GCN:
                arch_name = "3layergcn"
            else:
                arch_name = "mlp"
        else:
            print("num hidden not recognized")
            exit()
        model_path = (
            f"arch_{arch_name}-dataset_{dataset.name}-seed_{seed}-lr_{lr}-"
            f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}"
        )
        if test_dataset:
            model_path = (
                f"arch_{arch_name}-dataset_{dataset.name}_{test_dataset.name}-seed_{seed}-lr_{lr}-"
                f"hidden_size_{hidden_size}-num_hidden_{num_hidden}-dropout_{dropout}"
            )

        model_paths = [os.path.join(model_path, model_path + ".pth")]
    return model_paths

def parse_communication_gnn(model, mmlp_model_name, dataset):
    layers_size, _ = model.get_layers_size()
    embeddings_size = model.get_embeddings_size()
    gradients_size = model.get_gradients_size()
    
    comm_size = {}
    comm_size_with_server = {}

    with open(f"comm_results/{dataset}_{model.model_name}.json") as f:
        comm_data = json.load(f)
        
        for node in comm_data:
            l_size = 0
            for i in range(len(layers_size)):
                l_size += (comm_data[node]["layers"][i] * layers_size[i])
                
            e_size = 0
            for i in range(len(embeddings_size)):
                e_size += (comm_data[node]["embeddings"][i] * embeddings_size[i])
            
            g_size = 0
            for i in range(len(gradients_size)):
                g_size += (comm_data[node]["gradients"][i] * gradients_size[i])
            
            comm_size[node] = l_size + e_size + g_size
    
    with open(f"comm_results/{dataset}_{model.model_name}_with_server.json") as f:
        comm_data_with_server = json.load(f)
        
        for node in comm_data_with_server:
            l_size = 0
            for i in range(len(layers_size)):
                l_size += (comm_data_with_server[node]["layers"][i] * layers_size[i])
                
            comm_size_with_server[node] = l_size
            
    with open(f"comm_results/{dataset}_size_{mmlp_model_name}.json") as f:
        comm_size_mmlp = json.load(f)
        
    with open(f"comm_results/{dataset}_size_{mmlp_model_name}_with_server.json") as f:
        comm_size_with_server_mmlp = json.load(f)
    
    for node in comm_size:
        comm_size[node] += comm_size_with_server[node]
        comm_size_mmlp[node] += comm_size_with_server_mmlp[node]
    
    sorted_comm_size = sorted(comm_size.items(), key=lambda x:x[1], reverse=True)
    sorted_comm_size = dict(sorted_comm_size)
    
    sorted_comm_size_mmlp = OrderedDict((k, comm_size_mmlp[k]) for k in list(sorted_comm_size.keys()))
    with open(f"comm_results/{dataset}_sorted_size_{model.model_name}.json", "w") as f:
        json.dump(sorted_comm_size, f, indent = 4)
    
    with open(f"comm_results/{dataset}_sorted_size_mmlp_{model.model_name}.json", "w") as f:
        json.dump(sorted_comm_size_mmlp, f, indent=4) 
    
    plt.plot(list(sorted_comm_size.keys()), list(sorted_comm_size.values()), color='red')
    plt.plot(list(sorted_comm_size_mmlp.keys()), list(sorted_comm_size_mmlp.values()), color='blue')
    plt.savefig(f"comm_results/comm_size_{model.model_name}_{dataset}")
                
    f = csv.writer(open(f'comm_results/plot_values_{model.model_name}_{dataset}.csv', 'w', encoding='utf8'))
    f.writerow(['S_no', 'node'])
    
    for a, node in enumerate(sorted_comm_size):
        f.writerow([a, sorted_comm_size[node]])
        
    p = csv.writer(open(f'comm_results/plot_values_mmlp_{model.model_name}_{dataset}.csv', 'w', encoding='utf8'))
    p.writerow(['S_no', 'node'])
    
    for a, node in enumerate(sorted_comm_size_mmlp):
        p.writerow([a, sorted_comm_size_mmlp[node]])        
    return 
    
def parse_communication_mmlp(mmlp, dataset):
    models_size = mmlp.get_models_size()
    embeddings_size = mmlp.get_embeddings_size()
    
    comm_size_mmlp = {}
    comm_size_with_server_mmlp = {}
    
    with open(f"comm_results/{dataset}_{mmlp.model_name}.json") as f:
        comm_data = json.load(f)
        
        for node in comm_data:            
            e_size = 0
            for i in range(len(embeddings_size)):
                e_size += (comm_data[node]["embeddings"][i] * embeddings_size[i])
                
            comm_size_mmlp[node] = e_size
            
    with open(f"comm_results/{dataset}_{mmlp.model_name}_with_server.json") as f:           
        comm_data_with_server = json.load(f)
        
        for node in comm_data_with_server:
            l_size = 0
            for i in range(len(models_size)):
                l_size += (comm_data_with_server[node]["layers"][i] * models_size[i])
                
            comm_size_with_server_mmlp[node] = l_size
            
    with open(f"comm_results/{dataset}_size_{mmlp.model_name}.json", "w") as f:
        json.dump(comm_size_mmlp, f, indent = 4)
        
    with open(f"comm_results/{dataset}_size_{mmlp.model_name}_with_server.json", "w") as f:
        json.dump(comm_size_with_server_mmlp, f, indent = 4)
        
    return
     
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.max_validation_accuracy = 0
        
    def early_stop(self, validation_accuracy):
        if validation_accuracy > self.max_validation_accuracy:
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy <= (self.max_validation_accuracy - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
    
    def get_accuracy(self, current_accuracy, target_accuracy):
        if current_accuracy >= target_accuracy:
            self.counter += 1 
            if self.counter >= self.patience:
                return True
        elif current_accuracy < (target_accuracy - self.min_delta):
            self.counter = 0
            
        return False