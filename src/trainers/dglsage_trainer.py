import sys
sys.path.append("..")
from data import LoadData
from models.dglsage import DGLSage
from trainers.general_trainer import set_torch_seed, TrainStats
from torch.optim import SGD, Adam, lr_scheduler
from trainers.general_dgl_trainer import DGLTrainer
from tqdm import tqdm
import torch
import torch.nn as nn
import utils
import dgl
import numpy as np

def get_optimizer(model, run_config):
    if run_config.optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=run_config.learning_rate,
            momentum=run_config.momentum,
            weight_decay=run_config.weight_decay,
        )
    elif run_config.optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=run_config.learning_rate,
            weight_decay=run_config.weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer")


def get_scheduler(run_config, optimizer):
    def lr_lambda(current_step: int):
        if current_step < run_config.num_warmup_steps:
            return float(current_step) / float(max(1, run_config.num_warmup_steps))
        return max(
            0.0,
            float(run_config.num_epochs - current_step)
            / float(max(1, run_config.num_epochs - run_config.num_warmup_steps)),
        )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_dglsage_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    seeds=[1],
    sample_neighbors=False,
    feed_hidden_layer=False,
):
    print(
        "Training DGLSage dataset={}".format(dataset)
    )
    
    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    best_epochs = []
    
    for i in range(len(seeds)):
        set_torch_seed(seeds[i])
        rng = np.random.default_rng(seeds[i])
        
        data_loader = LoadData(
            dataset, 
            rng=rng,
            rng_seed=seeds[i],
            inductive = run_config.inductive
        )
        
        data = data_loader.train_data
        features = data_loader.features
        labels = data_loader.labels
        train_mask = data_loader.train_mask
        val_mask = data_loader.val_mask
        test_mask = data_loader.test_mask
        num_classes = data_loader.num_classes
        
        num_train_nodes = (data_loader.train_mask == True).sum().item()
        batch_size = run_config.batch_size if run_config.batch_size else num_train_nodes
        num_neighbors = [25, 25, 25] if sample_neighbors else [-1, -1]
        
        train_ids = (train_mask == True).nonzero(as_tuple=True)[0]
        val_ids = (val_mask == True).nonzero(as_tuple=True)[0]
        test_ids = (test_mask == True).nonzero(as_tuple=True)[0]
        
        sampler = dgl.dataloading.NeighborSampler(num_neighbors)
        train_dataloader = dgl.dataloading.DataLoader(
            data,
            train_ids,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        val_dataloader = dgl.dataloading.DataLoader(
            data,
            val_ids,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        model = DGLSage(
            input_size=features.size(1),
            hidden_size=run_config.hidden_size,            
            output_size=num_classes,
            num_hidden=run_config.num_hidden,
            dropout=run_config.dropout,
            device=device            
        )
        
        kwargs = {}
        trainer = DGLTrainer(model, rng, seed=seeds[i])
        
        val_loss, val_acc, best_epoch = trainer.train(
            dataset,
            train_dataloader,
            val_dataloader,
            device,
            run_config,
            kwargs=kwargs,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors
        )
                  
        test_dataloader = dgl.dataloading.DataLoader(
            data,
            test_ids,
            sampler,
            device=device,
            batch_size=test_mask.sum().item(),
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        test_loss, test_acc, f1_score, out_labels, logits = trainer.evaluate(
            test_dataloader, run_config, kwargs=kwargs
        )
        
        all_outputs.append((out_labels, logits))
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        f1_scores.append(f1_score)
        best_epochs.append(best_epoch)
        
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
        f1_scores
    )

    train_stats.print_stats()
    
    return train_stats
            
            
            
            
            