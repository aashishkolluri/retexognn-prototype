import os
import utils
import argparse
import numpy as np
import torch
import time
import torch.nn as nn
import models
from globals import MyGlobals
from trainers.general_trainer import RunConfig
from trainers.gcn_trainer import train_gcn_on_dataset
from trainers.mlp_trainer import train_mlp_on_dataset
from trainers.mmlp_trainer import train_mmlp_like_models
from trainers.sage_trainer import train_sage_on_dataset
from trainers.gat_trainer import train_gat_on_dataset


import pickle
import csv

def run_training(
    run_config, arch, dataset, device, dp=False, seeds=[1], test_dataset=None, feed_hidden_layer = False, sample_neighbors = False
):
    print("Train and evaluate arch {}; seeds {}".format(arch.name, seeds))
    start_time = time.time()
    if test_dataset:
        print("Transfer learning with separate test graph")

    if arch == utils.Architecture.GCN:
        train_stats = train_gcn_on_dataset(
            run_config, dataset, device, seeds=seeds, test_dataset=test_dataset, feed_hidden_layer=feed_hidden_layer, sample_neighbors=sample_neighbors
        )
    elif arch == utils.Architecture.MLP:
        train_stats = train_mlp_on_dataset(
            run_config,
            dataset,
            device,
            seeds=seeds,
            test_dataset=test_dataset,
        )
    elif arch == utils.Architecture.MMLP:
        train_stats = train_mmlp_like_models(
            run_config,
            dataset,
            device,
            seeds=seeds,
            test_dataset=test_dataset,
            feed_hidden_layer=feed_hidden_layer
        )
    elif arch in [utils.Architecture.MMLPGCN, utils.Architecture.MMLPSAGE, utils.Architecture.MMLPGAT]:
        train_stats = train_mmlp_like_models(
            run_config,
            dataset,
            device,
            arch,
            seeds=seeds,
            test_dataset=test_dataset,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
        )
    elif arch == utils.Architecture.GraphSAGE:
        train_stats = train_sage_on_dataset(
            run_config,
            dataset,
            device,
            seeds=seeds,
            test_dataset=None,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
        )
    elif arch == utils.Architecture.GAT:
        train_stats = train_gat_on_dataset(
            run_config,
            dataset,
            device,
            seeds=seeds,
            test_dataset=None,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
        )
    else:
        print("Arch {} not supported".format(arch))
        return None
    print(f"Training time: {time.time()-start_time}")
    return train_stats


# we get these args from argParser of the main function
def train(args):
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        # Run the train/attack/eval on the selected GPU id
        if torch.cuda.is_available:
            torch.cuda.set_device(args.cuda_id)
            print("Current CUDA device: {}".format(torch.cuda.current_device()))

    # define a run_config to train from the agruments of the args parser from trainer.py
    run_config = RunConfig(
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_each_epoch=False,
        save_epoch=args.save_epoch,
        weight_decay=MyGlobals.weight_decay,
        output_dir=os.path.join(args.outdir, "models"),
        nl=args.nl,
        hidden_size=args.hidden_size,
        attn_heads=args.attn_heads,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        inductive=args.inductive,
        calculate_communication=args.calculate_communication,
        parse_communication_results=args.parse_communication_results,
        early_stopping=args.early_stopping,
        diff_nei=args.diff_nei,
        frac=args.frac,
        hetero=args.hetero,
    )

    #generate array of random numbers as seeds if num_seeds > 1 else return [args.sample_seed]
    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)

    print("Running training")
    
    # run_training - return the training stats
    train_stats = run_training(
        run_config,
        args.arch,
        args.dataset,
        device,
        seeds=seeds,
        test_dataset=args.test_dataset,
        feed_hidden_layer=args.feed_hidden_layer,
        sample_neighbors=args.sample_neighbors,
    )

    if (
        args.arch == utils.Architecture.MMLP
        or args.arch == utils.Architecture.SimpleMMLP
    ):
        args.arch = args.arch.name + "_nl" + str(args.nl)

    dataset_name = str(args.dataset)

    if train_stats is not None:
        results_dict = {args.dataset: {}}
        results_dict[args.dataset][args.arch] = train_stats

        utils.save_results_pkl(
            results_dict, args.outdir, args.arch, dataset_name, run_config,
            feed_hidden_layer=args.feed_hidden_layer, sample_neighbors=args.sample_neighbors,
            inductive=args.inductive 
        )

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--parse_best_config", action="store_true", default=False)
    parser.add_argument("--parse_best_hyperparameters", action="store_true", default=False)
    
    parser.add_argument("--best_config_file", type=str, default="./best_config.pkl")
    parser.add_argument("--best_acc_file", type=str, default="./best_config.csv")
    parser.add_argument("--best_hyperparams_file", type=str, default="./best_hyperparameters.csv")
    
    parser.add_argument("--hetero", action="store_true", default=False)
    
    parser.add_argument(
        "--dataset",
        type=utils.Dataset,
        choices=utils.Dataset,
        default=utils.Dataset.Cora,
    )
    parser.add_argument(
        "--inductive",
        default=False,
        action="store_true",
        help="Whether to use inductive setting",
    )
    parser.add_argument(
        "--arch",
        type=utils.Architecture,
        choices=utils.Architecture,
        default=utils.Architecture.MMLP,
        help="Type of architecture to train",
    )
    parser.add_argument(
        "--nl", type=int, default=MyGlobals.nl, help="Number of stacked models"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=MyGlobals.num_seeds,
        help="Run over num_seeds seeds",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=MyGlobals.sample_seed,
        help="Run for this seed",
    )
    parser.add_argument("--cuda_id", type=int, default=MyGlobals.cuda_id)
    parser.add_argument("--no_cuda", action="store_true", default=False)

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=MyGlobals.hidden_size,
        help="Size of the first hidden layer",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=MyGlobals.num_hidden,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--attn_heads",
        type=int,
        default=MyGlobals.attn_heads,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=MyGlobals.RESULTDIR,
        help="Directory to save the models and results",
    )
    parser.add_argument(
        "--test_dataset", type=utils.Dataset, choices=utils.Dataset, default=None
    )
    
    parser.add_argument(
        "--batch_size", type=int, help="number of nodes sampled from train and val each for trainer"
    )
    
    parser.add_argument(
        "--frac", type=float, help="fraction of neighbors to sample for every mlp in retexo models"
    )
    
    parser.add_argument(
        "--sample_neighbors", action="store_true", default=False,
    )
    parser.add_argument(
        "--feed_hidden_layer", action="store_true", default=False,
    )
    
    parser.add_argument(
        "--calculate_communication", action="store_true", default=False,
    )
    
    parser.add_argument(
        "--parse_communication_results", action="store_true", default=False,
    )
    
    parser.add_argument(
        "--early_stopping", action="store_true", default=False,
    )
    
    parser.add_argument(
        "--diff_nei", action="store_true", default=False,
    )

    # Train model commands
    # Should add more -- perhaps set a config file to make it easier to set all of these parameters
    # used to add multiple parser depending upon the argument given for subparser
    
    subparsers = parser.add_subparsers(help="sub-command help")

    # defining parser for train 
    train_parser = subparsers.add_parser("train", help="train sub-menu help")
    train_parser.add_argument(
        "--lr", type=float, default=MyGlobals.lr, help="Learning rate"
    )
    train_parser.add_argument("--num_epochs", type=int, default=MyGlobals.num_epochs)
    train_parser.add_argument(
        "--save_epoch",
        type=int,
        default=MyGlobals.save_epoch,
        help="Save at every save_epoch",
    )
    train_parser.add_argument("--dropout", type=float, default=MyGlobals.dropout)
    train_parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Store the results in a pkl in args.outdir",
    )
    train_parser.set_defaults(func=train)
    
    
    args = parser.parse_args()

    if args.parse_best_config:
        parse_best_config(args.best_config_file, args.best_acc_file)
        return
    elif args.parse_best_hyperparameters:
        parse_best_hyperparameters(args.best_config_file, args.best_acc_file)
        return 
    
    args.func(args)


def parse_best_config(best_config_file, best_acc_file):
    with open(best_config_file, 'rb') as f:
        data = pickle.load(f)
        configs_json = data
        
        f = csv.writer(open(best_acc_file, 'w', encoding='utf8'))
        
        
        f.writerow(["S_no","cora", "citeseer", "pubmed", "facebook_page", "lastfm_asia"])
        archs = ['mlp', 'gcn', 'mmlp_gcn', 'graphSAGE', 'mmlp_sage', 'gat', 'mmlp_gat']
        
        for arch in archs:
            f.writerow([arch, 
                       configs_json['cora'][arch][-1],
                       configs_json['citeseer'][arch][-1],
                       configs_json['pubmed'][arch][-1],
                       configs_json['facebook_page'][arch][-1],
                       configs_json['lastfm_asia'][arch][-1]])
        
def parse_best_hyperparameters(best_config_file, best_hyperparameters_file):
    with open(best_config_file, 'rb') as f:
        data = pickle.load(f)
        configs_json = data 
        
        fl = csv.writer(open(best_hyperparameters_file, 'w', encoding='utf8'))

        fl.writerow(["S_no", "cora", "citeseer", "pubmed", "facebook_page", "lastfm_asia"])      
        archs = ['mlp', 'gcn', 'mmlp_gcn', 'graphSAGE', 'mmlp_sage', 'gat', 'mmlp_gat']
        
        for arch in archs:
            fl.writerow([arch,
                        configs_json['cora'][arch][0], 
                        configs_json['citeseer'][arch][0],
                        configs_json['pubmed'][arch][0],
                        configs_json['facebook_page'][arch][0],
                        configs_json['lastfm_asia'][arch][0]])   
    
if __name__ == "__main__":
    main()
