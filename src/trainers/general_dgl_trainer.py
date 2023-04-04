import copy
import sys
sys.path.append("..")
import os 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from sklearn.metrics import f1_score
from tqdm import tqdm 

import utils
from globals import MyGlobals
from utils import EarlyStopper

def lk_f1_score(preds, labels):
    return (
        f1_score(labels.cpu(), preds.detach().cpu(), average="micro"),
        f1_score(labels.cpu(), preds.detach().cpu(), average="macro"),
        f1_score(labels.cpu(), preds.detach().cpu(), average="weighted"),
    )
    
class DGLTrainer:
    def __init__(self, model, rng, seed):
        self.model = model
        self.rng = rng
        self.seed = seed
        self.out_labels = []
        
    def get_optimizer(self, run_config):
        if run_config.optimizer_name == "sgd":
            return SGD(
                self.model.parameters(),
                lr=run_config.learning_rate,
                momentum=run_config.momentum,
                weight_decay=run_config.weight_decay,
            )
        elif run_config.optimizer_name == "adam":
            return Adam(
                self.model.parameters(),
                lr=run_config.learning_rate,
                weight_decay=run_config.weight_decay,
            )
        else:
            raise ValueError("Unknown optimizer")

    def get_scheduler(self, run_config, optimizer):
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
    
    def get_early_stopper(self, run_config):
        return EarlyStopper(patience=run_config.early_stopping_patience, min_delta=run_config.early_stopping_min_delta) 
    
    def train(
        self,
        dataset,
        train_loader,
        val_loader,
        device,
        run_config,
        log=False,
        kwargs={},
        feed_hidden_layer=False,
        sample_neighbors=False,
        store_result=True,
    ):

        self.model = self.model.to(device)
        optimizer = self.get_optimizer(run_config)
        scheduler = self.get_scheduler(run_config, optimizer)
        early_stopper = self.get_early_stopper(run_config)
        
        if log:
            print("Training Started:")
            print(f"\tNum Epochs = {run_config.num_epochs}")
            print(f"\tSave Each Epoch = {run_config.save_each_epoch}")
            
        best_loss, best_model_accuracy = float("inf"), 0
        best_model_state_dict = None
        best_epoch = None
        dir_name = utils.get_folder_name(
                    run_config,
                    dataset,
                    self.model.model_name,
                    self.seed,
                    feed_hidden_layer=feed_hidden_layer,
                    sample_neighbors=sample_neighbors,
                    inductive=run_config.inductive,
                )
        best_output_dir = os.path.join(run_config.output_dir, dir_name)            
        
        train_iterator = tqdm(range(0, int(run_config.num_epochs)), desc="Epoch")
        
        for epoch in train_iterator:
            self.model.train()
            
            total_loss = total_correct = total_examples = 0
            
            for i, (input_nodes, output_nodes, mfgs) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']
                current_batch_size = labels.size()[0]
                predictions = self.model(mfgs, inputs, **kwargs)
                
                loss = nn.CrossEntropyLoss()(predictions, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += float(loss) * current_batch_size
                total_correct += int((predictions.argmax(dim=-1) == labels).sum())
                total_examples += current_batch_size
                
            loss = total_loss / total_examples
            acc = total_correct / total_examples
            
            val_loss, val_accuracy, _, _, _ = self.evaluate(
                val_loader, run_config, kwargs=kwargs
            )
            
            train_iterator.set_description(
                f"Training Loss = {loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val Accuracy = {val_accuracy:.2f}"
            )
            
            save_best_model = val_loss < best_loss
            if save_best_model:
                best_epoch = epoch + 1 
                best_loss = val_loss
                best_model_accuracy = val_accuracy
                del best_model_state_dict
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                
            if run_config.early_stopping and early_stopper.early_stop(val_accuracy):
                break
        
        if log:
            print(
                f"Best model val CE loss = {best_loss:.4f}, "
                f"Best model val accuracy = {best_model_accuracy:.2f}"
            )
            
        self.model.load_state_dict(best_model_state_dict)
        if best_output_dir and store_result:
            self.model.save(best_output_dir)
            print("Best saved at {}".format(best_output_dir))
            
        return best_loss, best_model_accuracy, best_epoch
            
    def evaluate(self, neigh_loader, run_config, kwargs={}):
        self.model.eval()
        total_loss = total_correct = total_examples = 0
        total_f1_score_0 = total_f1_score_1 = total_f1_score_2 = 0
        
        out_labels = []
        logits = []
        
        for i, (input_nodes, output_nodes, mfgs) in enumerate(neigh_loader):
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            current_batch_size = labels.size()[0]
            predictions = self.model(mfgs, inputs, **kwargs)
            
            logits.append(predictions.cpu().detach().numpy())
            loss = nn.CrossEntropyLoss()(predictions, labels)
            
            total_loss += float(loss) * current_batch_size
            total_correct += int((predictions.argmax(dim=-1) == labels).sum())
            total_examples += current_batch_size
            
            predicted_labels = predictions.argmax(dim=-1)
            out_labels.append(predicted_labels.cpu().detach().numpy())
            
            f1_score_0, f1_score_1, f1_score_2= lk_f1_score(predicted_labels, labels)
            
            total_f1_score_0 += float(f1_score_0) * current_batch_size
            total_f1_score_1 += float(f1_score_1) * current_batch_size
            total_f1_score_2 += float(f1_score_2) * current_batch_size
        
        loss = total_loss / total_examples
        acc = total_correct / total_examples
        my_f1_score = (total_f1_score_0 / total_examples, total_f1_score_1 / total_examples, total_f1_score_2 / total_examples)
        
        return loss, acc, my_f1_score, out_labels, logits
    
    
            
            
            