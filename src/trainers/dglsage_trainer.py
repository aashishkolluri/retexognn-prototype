import sys
sys.path.append("..")
from data import LoadData
from models.DGLSage import DGLSage
from trainers.general_trainer import set_torch_seed
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
from sklearn import metrics
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


def calc_f1(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    a = nn.Softmax(dim=1)
    y_pred = a(y_pred)
    y_pred = y_pred.detach().cpu().numpy()
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    
    return metrics.f1_score(y_true, y_pred, average="micro")

def train_dglsage_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    seeds=[1],
    sample_neighbors=False
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
        num_neighbors = [25, 25] if sample_neighbors else [-1, -1]
        
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
        
        optimizer = get_optimizer(model, run_config)
        scheduler = get_scheduler(run_config, optimizer)
        
        train_iterator = tqdm(range(0, int(run_config.num_epochs)), desc="Epoch")
        
        for epoch in train_iterator:
            model.train()
            
            total_train_loss = total_val_loss = 0
            total_train_correct = total_val_correct = 0
            total_train_examples = total_val_examples = 0
            num_batch = len(train_dataloader)
            
            for i, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']
                labels = labels.to(torch.float32)
                batch_size = labels.size()[0]
                predictions = model(mfgs, inputs)
                
                loss = nn.CrossEntropyLoss()(predictions, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
              
                total_train_loss += float(loss) * batch_size
                total_train_correct += int((predictions.argmax(dim=-1) == labels).sum())
                total_train_examples += batch_size
            
            model.eval()
            
            for i, (input_nodes, output_nodes, mfgs) in enumerate(val_dataloader):
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']
                labels = labels.to(torch.float32)
                batch_size = labels.size()[0]
                
                with torch.no_grad():
                    predictions = model(mfgs, inputs)
                    loss = nn.CrossEntropyLoss()(predictions, labels)
                
                total_val_loss += float(loss) * batch_size
                total_val_correct += int((predictions.argmax(dim=-1) == labels).sum())
                total_val_examples += batch_size

            train_loss = total_train_loss / total_train_examples
            val_loss = total_val_loss / total_val_examples
            val_acc = total_val_correct / total_val_examples
            
            train_iterator.set_description(
                f"Training loss = {train_loss:.4f}, "
                f"val loss = {val_loss: .4f}, "
                f"val accuracy = {val_acc: .2f}"
            )
            
        # y_test = labels[test_mask]
        # y_test = y_test.to(torch.float32)
        # y_hat_test = model(data, features)[test_mask]
        
        # test_loss = nn.BCEWithLogitsLoss()(y_hat_test, y_test)
        # f1_micro_test = calc_f1(y_test, y_hat_test)
        
        # print(f"Test Loss {test_loss.item()}")
        # print(f"Test Accuracy {f1_micro_test}")

            
            
            
            
            