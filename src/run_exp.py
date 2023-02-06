import multiprocessing
import argparse
from sys import stdout
import numpy as np
import shutil
import torch
from pathlib import Path
import os
from trainers.general_trainer import RunConfig
import utils
import main
from itertools import product
import pickle as pkl
from trainers.general_trainer import TrainStats

import io


def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc
    return lambda b: torch.load(io.BytesIO(b), map_location=map_loc)


class MappedUnpickler(pkl.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

    def __init__(self, *args, map_location="cpu", **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return fix(self._map_location)
        else:
            return super().find_class(module, name)


def mapped_loads(s, map_location="cpu"):
    bs = io.BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()


def read_best_config(out_path):
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            best_config = pkl.load(f)

        return best_config


def parse_best_config_from_dir(outdir, config_name):
    filenames = os.listdir(outdir)
    best_config = {}

    for filename in filenames:
        if os.path.isdir(filename):
            continue
        if not filename.endswith(".pkl"):
            continue

        with open(os.path.join(outdir, filename), "rb") as f:
            content = f.read()
            results_dict = mapped_loads(content)

        for dataset in list(results_dict):  
            results_dict[dataset.value] = results_dict[dataset]
            del results_dict[dataset]
            dataset = dataset.value  
            for arch in list(results_dict[dataset]):
                results_dict[dataset][arch.value] = results_dict[dataset][arch]
                del results_dict[dataset][arch]
                arch = arch.value
                for config in results_dict[dataset][arch]:
                    train_stats = results_dict[dataset][arch][config]
                    best_val_loss, best_val_acc = train_stats.get_best_avg_val()
                    best_test_loss, best_test_acc = train_stats.get_best_avg_test()
                    if not dataset in best_config:
                        best_config[dataset] = {
                            arch: [config, best_val_loss, best_val_acc, best_test_loss, best_test_acc]
                        }
                    elif not arch in best_config[dataset]:
                        best_config[dataset][arch] = [
                            config,
                            best_val_loss,
                            best_val_acc,
                            best_test_loss,
                            best_test_acc
                        ]
                    else:
                        if best_val_loss < best_config[dataset][arch][1]:
                            best_config[dataset][arch] = [
                                config,
                                best_val_loss,
                                best_val_acc,
                                best_test_loss,
                                best_test_acc                                   
                                ]

    with open(config_name, "wb") as f:
        pkl.dump(best_config, f)

    return best_config


def create_todos(datasets,architecture_names, configs, TODOS_DIR):
    todos = []
    tasks_to_run = list(product(datasets, architecture_names, configs))
    if not os.path.exists(TODOS_DIR):
        for task in tasks_to_run:
            dataset = task[0]
            arch = task[1]
            lr = task[2][0]
            hidden_size = task[2][1]
            num_hidden = task[2][2]
            dropout = task[2][3]
            feed_hidden_layer = task[2][4]
            sample_neighbors = task[2][5]
            inductive = task[2][6]


            todo = f"{arch}-{dataset.name}-{lr}-{hidden_size}-{num_hidden}-{dropout}"
                
            if feed_hidden_layer:
                todo += f"-feed_hidden_layer"
                
            if sample_neighbors:
                todo += f"-sample_neighbors"
                    
            if inductive:
                todo += f"-inductive"
                    
            todos.append(todo)
        
        Path(TODOS_DIR).mkdir(exist_ok=True, parents=True)
        for todo in todos:
            if not os.path.exists(os.path.join(TODOS_DIR, todo)):
                f = open(os.path.join(TODOS_DIR, todo), "w")
                f.close()
    try:
        os.mkdir(TODOS_DIR)
    except Exception:
        print("Cannot create {}".format(TODOS_DIR))
        pass


def create_best_todos(datasets, architecture_names, best_configs, TODOS_DIR):
    todos = []
    tasks_to_run = list(product(datasets, architecture_names))
    if not os.path.exists(TODOS_DIR):
        for task in tasks_to_run:
            dataset = task[0]
            arch = task[1]
            # Inconsistent but :(
            if arch in ("GCN", "MLP"):
                arch = utils.Architecture[arch]
                arch_name = arch.name
            else:
                arch_name = arch

            if dataset not in best_configs:
                print("{} missing".format(dataset))
                continue
            else:
                lr = best_configs[dataset][0][arch][2].run_config.learning_rate
                hidden_size = best_configs[dataset][0][arch][2].run_config.hidden_size
                num_hidden = best_configs[dataset][0][arch][2].run_config.num_hidden
                dropout = best_configs[dataset][0][arch][2].run_config.dropout

                todos.append(
                    f"{arch_name}-{dataset.name}-{lr}-{hidden_size}-{num_hidden}-{dropout}"
                )

        Path(TODOS_DIR).mkdir(exist_ok=True, parents=True)
        for todo in todos:
            if not os.path.exists(os.path.join(TODOS_DIR, todo)):
                f = open(os.path.join(TODOS_DIR, todo), "w")
                f.close()
    try:
        os.mkdir(TODOS_DIR)
    except Exception:
        print("Cannot create {}".format(TODOS_DIR))
        pass


def train_for_config(device, outdir, num_epochs, seeds, TODOS_DIR, DONE_DIR):

    Path(outdir).mkdir(exist_ok=True, parents=True)
    Path(DONE_DIR).mkdir(exist_ok=True, parents=True)
    todos_files = os.listdir(TODOS_DIR)

    results_dict = {}
    while todos_files:
        results_dict = {}
        todo = np.random.choice(todos_files)
        todos_files.remove(todo)
        todo_path = os.path.join(TODOS_DIR, todo)
        working_path = os.path.join(DONE_DIR, todo)
        print("Working on {}".format(working_path))
        try:
            shutil.move(todo_path, working_path)
        except IOError:
            continue

        params = todo.split("-")
        arch = utils.Architecture[params[0].split("_")[0]]


        dataset = utils.Dataset[params[1]]
        test_dataset = None

        lr = float(params[2])
        hidden_size = int(params[3])
        num_hidden = int(params[4])
        dropout = float(params[5])
        feed_hidden_layer = False
        sample_neighbors = False
        inductive = False
        
        try:
            if params[6] == "feed_hidden_layer":
                feed_hidden_layer = True
            elif params[6] == "sample_neighbors":
                sample_neighbors = True
            elif params[6] == "inductive":
                inductive = True
        except:
            pass
        
        try:
            if params[7] == "sample_neighbors":
                sample_neighbors = True
            elif params[7] == "feed_hidden_layer":
                feed_hidden_layer = True
            elif params[7] == "inductive":
                inductive = True
        except:
            pass
        
        try:
            if params[8] == "sample_neighbors":
                sample_neighbors = True
            elif params[8] == "feed_hidden_layer":
                feed_hidden_layer = True
            elif params[8] == "inductive":
                inductive = True
        except:
            pass
        
        if arch == utils.Architecture.MLP and num_hidden == 1:
            continue

        if arch in [utils.Architecture.MMLP, utils.Architecture.MMLPSAGE, utils.Architecture.MMLPGCN, utils.Architecture.MMLPGAT] or arch == utils.Architecture.SimpleMMLP:
            nl = int(params[0].split("_")[1][2:])
            print(
                "Running dataset={} arch={} nl={}".format(dataset, arch, nl)
            )
        else:
            nl = 1
            print("Running dataset={} arch={}".format(dataset, arch))

        run_config = RunConfig(
            learning_rate=lr,
            num_epochs=num_epochs,
            save_each_epoch=False,
            save_epoch=100,
            weight_decay=5e-4,
            output_dir=os.path.join(outdir, "models"),
            nl=nl,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout=dropout,
            batch_size=512,
            attn_heads=8,
            inductive=inductive
        )
        
        print(feed_hidden_layer)
        print(sample_neighbors)
        print(inductive)
        train_stats = main.run_training(
            run_config,
            arch,
            dataset,
            device,
            seeds=seeds,
            test_dataset=test_dataset,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
        )
        
        if arch == utils.Architecture.MMLP or arch == utils.Architecture.SimpleMMLP:
            arch = arch.name + "_nl" + str(nl)


        dataset_name = str(dataset)

        config = f"{lr}-{hidden_size}-{num_hidden}-{dropout}"
        curr_dict = {dataset:  {arch: {config: train_stats}}}

        if not dataset in results_dict:
            results_dict = curr_dict
        elif not arch in results_dict[dataset]:
            results_dict[dataset][arch] = curr_dict[dataset][arch]
        else:
            results_dict[dataset][arch][config] = curr_dict[dataset][arch][
                config
            ]

        utils.save_results_pkl(curr_dict, outdir, arch, dataset_name, run_config, feed_hidden_layer=feed_hidden_layer, sample_neighbors=sample_neighbors, inductive=inductive)
    return results_dict


def run_all_commands(device, args):
    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)

    if args.hyperparameters == True and args.best_config_file is not None:
        print("You either do hyperparameter search or run for best config")
        exit()
    if args.parse_config_dir:
        print("Parsing configs and writing them to {}".format(args.best_config_file))
        parse_best_config_from_dir(args.parse_config_dir, args.best_config_file)
        return ()

    print("We run for these seeds {}".format(seeds))
    TODOS_DIR = os.path.join(args.todos_dir, "working")
    DONE_DIR = os.path.join(args.todos_dir, "done")

    architecture_names = []

    # archs = [utils.Architecture.MLP, utils.Architecture.GCN, utils.Architecture.MMLPGCN, utils.Architecture.GraphSAGE, utils.Architecture.MMLPSAGE, utils.Architecture.GAT, utils.Architecture.MMLPGAT]
    
    # archs = [utils.Architecture.GAT, utils.Architecture.MMLPGAT]
    archs = [utils.Architecture.MLP]
    for arch in archs:
        # Skip SimpleMMLP for experiments
        if arch == utils.Architecture.SimpleMMLP:
            continue
        if arch == utils.Architecture.TwoLayerGCN:
            continue
        if "mmlp" in arch.value:
            arch_name = arch.name + "_nl" + str(2)
            architecture_names.append(arch_name)
        else:
            arch_name = arch.name
            architecture_names.append(arch_name)



    if args.datasets:
        datasets = [utils.Dataset[x] for x in args.datasets.split(",")]
        sample_types = ["balanced"]

    else:
        datasets = [
            utils.Dataset.Cora,
            utils.Dataset.CiteSeer,
            utils.Dataset.PubMed,
            utils.Dataset.facebook_page,
            utils.Dataset.LastFM,
            ]
    print("TODOS DIR {} DONE DIR {}".format(TODOS_DIR, DONE_DIR))
    # training parameters, there is no batch size as we use the whole set in each iteration
    
    if args.hyperparameters:
        learning_rates = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
        hidden_sizes_1 = [64, 128, 256]
        num_hidden = [2]
        dropout_rates = [0.0]
        feed_hidden_layer = [False]
        sample_neighbors = [True]
        inductive = [True]
        configs = list(
            product(learning_rates, hidden_sizes_1, num_hidden, dropout_rates, feed_hidden_layer, sample_neighbors, inductive)
        )
        create_todos(datasets, architecture_names, configs, TODOS_DIR)
        if args.only_create_todos:
            return
    elif args.best_config_file:
        configs = read_best_config(args.best_config_file)
        create_best_todos(datasets, architecture_names, configs, TODOS_DIR)
        if args.only_create_todos:
            return
    else:
        print(args.datasets)
        # The hyperparameters in the LinkTeller paper give better performance
        if datasets == [utils.Dataset.Flickr]:
            learning_rates = [0.0005]
            hidden_sizes_1 = [256]
            num_hidden = [2]
            dropout_rates = [0.2]
        else:
            learning_rates = [0.01]
            hidden_sizes_1 = [16]
            num_hidden = [2]
            dropout_rates = [0.5]
        configs = list(
            product(learning_rates, hidden_sizes_1, num_hidden, dropout_rates)
        )
        create_todos(datasets, architecture_names, configs, TODOS_DIR)
        if args.only_create_todos:
            return

    if args.command == "train":
        train_for_config(
            device, args.outdir, args.num_epochs, seeds, TODOS_DIR, DONE_DIR
        )
    else:
        print(args.command)
        exit()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=30)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_stacked", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument(
        "--outdir",
        type=str,
        default="../data-test",
        help="Directory to save the models and results",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--inductive", default=False, action="store_true")
    parser.add_argument(
        "--command", type=str, choices=["train"], default="train"
    )
    parser.add_argument("--hyperparameters", action="store_true", default=False)
    parser.add_argument("--parse_config_dir", type=str, default=None)
    parser.add_argument("--best_config_file", type=str, default=None)
    parser.add_argument("--distribute", action="store_true", default=False)
    parser.add_argument("--todos_dir", type=str, default=None)
    parser.add_argument("--only_create_todos", action="store_true", default=False)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Specific datasets separated by comma",
    )
    parser.add_argument("--feed_hidden_layer", action="store_true", default=False)
    parser.add_argument("--sample_neighbors", action="store_true", default=False)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run the train/attack/eval on the selected GPU id
    if torch.cuda.is_available:
        torch.cuda.set_device(args.cuda_id)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    run_all_commands(device, args)


if __name__ == "__main__":
    run()
