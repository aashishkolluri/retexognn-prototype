import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

import utils
from globals import MyGlobals
from utils import EarlyStopper


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class RunConfig:  # Later overwritten in the main function. Only declaration and default initailization here.
    learning_rate: float = MyGlobals.lr
    num_epochs: int = MyGlobals.num_epochs
    weight_decay: float = MyGlobals.weight_decay
    num_warmup_steps: int = 0
    save_each_epoch: bool = False
    save_epoch: int = MyGlobals.save_epoch
    output_dir: str = "."
    hidden_size: int = MyGlobals.hidden_size
    num_hidden: int = MyGlobals.num_hidden
    attn_heads: int = MyGlobals.attn_heads
    dropout: float = MyGlobals.dropout
    nl: int = MyGlobals.nl
    momentum: float = MyGlobals.momentum
    batch_size: int = None
    optimizer_name: str = MyGlobals.optimizer
    early_stopping: bool = False
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.001
    inductive: bool = False
    calculate_communication: bool = False
    parse_communication_results: bool = False
    diff_nei: bool = False
    frac: float = 1.0
    hetero: bool = False


class TrainStats(object):
    """
    A class that encapsulates stats about the trained model
    """

    def __init__(
        self,
        run_config,
        dataset,
        model_name,
        all_outs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores=None,
        test_dataset=None,
    ):
        super(TrainStats, self).__init__
        self.all_outs = all_outs
        self.run_config = run_config
        self.validation = {"loss": val_losses, "acc": val_accuracies}
        self.testing = {"loss": test_losses, "acc": test_accuracies}
        self.best_epochs = best_epochs
        self.dataset = dataset
        self.model_name = model_name
        self.f1_scores = f1_scores
        self.rare_f1_scores = rare_f1_scores
        self.seeds = seeds
        self.test_dataset = test_dataset

    def get_best_avg_val(self):
        """
        Returns the best model for the seed
        """
        val_loss = np.mean(self.validation["loss"])
        val_acc = np.mean(self.validation["acc"])
        
        val_loss_std = np.std(self.validation["loss"])
        val_acc_std = np.std(self.validation["acc"])
        
        return f"{val_loss: .3f} +- {val_loss_std: .3f}", f"{val_acc: .3f} +- {val_acc_std: .3f}"
    
    def get_best_avg_test(self):
        test_loss = np.mean(self.testing["loss"])
        test_acc = np.mean(self.testing["acc"])
        
        test_loss_std = np.std(self.testing["loss"])
        test_acc_std = np.std(self.testing["acc"])
        
        return f"{test_loss: .3f} +- {test_loss_std: .3f}", f"{test_acc: .3f} +- {test_acc_std: .3f}"

    def print_stats(self):
        print("Best epochs: {}".format(self.best_epochs))
        print(
            "Best val_loss, val_acc {} {}".format(
                self.validation["loss"], self.validation["acc"]
            )
        )
        f1_scores = np.stack(self.f1_scores, axis=1)
        print(
            f"\nPerformance on {self.dataset.name}:\n- "
            f"test accuracy = {np.mean(self.testing['acc']):.3f} +-"
            f"{np.std(self.testing['acc']):.3f}\n- "
            f"micro f1 score = {np.mean(f1_scores[0]):.3f}  +-"
            f"{np.std(f1_scores[0]):.3f}\n- "
            f"macro f1 score = {np.mean(f1_scores[1]):.3f} +-"
            f"{np.std(f1_scores[1]):.3f}\n- "
            f"weighted f1 score = {np.mean(f1_scores[2]):.3f} +-"
            f"{np.std(f1_scores[2]):.3f}\n"
        )
        if self.rare_f1_scores:
            rare_f1_scores = np.stack(self.rare_f1_scores, axis=1)
            print(
                f"\nPerformance on {self.test_dataset.name}:\n- "
                f"test accuracy = {np.mean(self.testing['acc']):.3f} +-"
                f"{np.std(self.testing['acc']):.3f}\n- "
                f"F1 score = {np.mean(rare_f1_scores[0]):.3f}  +-"
                f"{np.std(rare_f1_scores[0]):.3f}\n- "
                f"Precision = {np.mean(rare_f1_scores[1]):.3f} +-"
                f"{np.std(rare_f1_scores[1]):.3f}\n- "
                f"Recall = {np.mean(rare_f1_scores[2]):.3f} +-"
                f"{np.std(rare_f1_scores[2]):.3f}\n"
                f"AP = {np.mean(rare_f1_scores[3]):.3f} +-"
                f"{np.std(rare_f1_scores[3]):.3f}\n"
            )

class Trainer:
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
        test_dataset=None,
        kwargs={}, # keyword arguments for the model
        feed_hidden_layer=False,
        sample_neighbors=False,
        store_result=True,
    ):
        """
        Return the loss and accuracy on the validation set if validation is available.
        Otherwise, the loss and accuracy on training.
        """
        self.model = self.model.to(device)

        optimizer = self.get_optimizer(run_config)
        scheduler = self.get_scheduler(run_config, optimizer)
        early_stopper = self.get_early_stopper(run_config)

        if log:
            print("Training started:")
            print(f"\tNum Epochs = {run_config.num_epochs}")
            print(f"\tSave each epoch = {run_config.save_each_epoch}")

        best_loss, best_model_accuracy = float("inf"), 0
        best_model_state_dict = None
        best_epoch = None
        dir_name = utils.get_folder_name(
                    run_config,
                    dataset,
                    self.model.model_name,
                    self.seed,
                    test_dataset=test_dataset,
                    feed_hidden_layer=feed_hidden_layer,
                    sample_neighbors=sample_neighbors,
                    inductive=run_config.inductive,
                )
        best_output_dir = os.path.join(run_config.output_dir, dir_name)

        train_iterator = tqdm(range(0, int(run_config.num_epochs)), desc="Epoch")
        if run_config.calculate_communication:
            comm_stats = kwargs["comm_stats"]
            comm_stats_with_server = kwargs["comm_stats_with_server"]
            is_mmlp = kwargs["is_mmlp"]
            train_model = kwargs["train_model"]
        
        for epoch in train_iterator:
            self.model.train()
            
            total_loss = total_correct = total_examples = 0
            num_batch = len(train_loader)
            
            for i, batch in enumerate(train_loader):
                if i % num_batch == 0:
                    if run_config.calculate_communication:
                        self.calculate_communication(batch, comm_stats, comm_stats_with_server, is_mmlp, train_model)     
                    optimizer.zero_grad()
                    
                    kwargs["edge_index"] = batch.edge_index.to(device)
                
                    y = batch.y[:batch.batch_size]
                    y_hat = self.model(batch.x , **kwargs)[:batch.batch_size]
                    loss = nn.CrossEntropyLoss()(y_hat, y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += float(loss) * batch.batch_size
                    total_correct += int((y_hat.argmax(dim=-1) == y).sum())
                    total_examples += batch.batch_size

            loss = total_loss / total_examples
            acc = total_correct / total_examples            
            
            val_loss, val_accuracy, _, _, _, _ = self.evaluate(
                val_loader, run_config, kwargs=kwargs
            )

            train_iterator.set_description(
                f"Training loss = {loss:.4f}, "
                f"val loss = {val_loss:.4f}, "
                f"val accuracy = {val_accuracy:.2f}"
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
                f"best model val accuracy = {best_model_accuracy:.2f}"
            )

        self.model.load_state_dict(best_model_state_dict)
        
        if best_output_dir and store_result:
            self.model.save(best_output_dir)
            print("Best saved at {}".format(best_output_dir))
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_loss, best_model_accuracy, best_epoch

    def evaluate(self, neigh_loader, run_config, is_rare=False, kwargs={}):
        self.model.eval()
        total_loss = total_correct = total_examples = total_rare_f1_score = 0
        total_f1_score_0 = total_f1_score_1 = total_f1_score_2 = 0

        out_labels = []
        logits = []
        
        num_batches = len(neigh_loader)
        
        if run_config.calculate_communication:
            comm_stats = kwargs["comm_stats"]
            comm_stats_with_server = kwargs["comm_stats_with_server"]
            is_mmlp = kwargs["is_mmlp"]
            train_model = kwargs["train_model"]
        
        for i, batch in enumerate(neigh_loader):
            if i % num_batches == 0:
                if run_config.calculate_communication:
                    self.calculate_communication(batch, comm_stats, comm_stats_with_server, is_mmlp, train_model) 
                    
                y = batch.y[:batch.batch_size]
                
                kwargs["edge_index"] = batch.edge_index.to(batch.x.device)
                y_hat = self.model(batch.x, **kwargs)[:batch.batch_size]

                logits.append(y_hat.cpu().detach().numpy())
                loss = nn.CrossEntropyLoss()(y_hat, y)
                
                total_loss += float(loss) * batch.batch_size
                total_correct += int((y_hat.argmax(dim=-1) == y).sum())
                total_examples += batch.batch_size

                preds = F.softmax(y_hat, dim=1)
                predicted_labels = torch.max(preds, dim=1).indices[:]
                out_labels.append(predicted_labels[1].cpu().detach().numpy() if len(predicted_labels) > 1 else predicted_labels[0].cpu().detach().numpy())
                
                true_labels = y
                
                f1_score_0, f1_score_1, f1_score_2 = lk_f1_score(predicted_labels, true_labels)
                total_f1_score_0 += float(f1_score_0) * batch.batch_size
                total_f1_score_1 += float(f1_score_1) * batch.batch_size
                total_f1_score_2 += float(f1_score_2) * batch.batch_size
                
                if is_rare:
                    total_rare_f1_score += rare_class_f1(y_hat, true_labels)[0] * batch.batch_size
            
        loss = total_loss / total_examples
        acc = total_correct / total_examples
        my_f1_score = (total_f1_score_0 / total_examples, total_f1_score_1 / total_examples, total_f1_score_2 / total_examples)
        rare_f1_score = total_rare_f1_score / total_examples
        
        return loss, acc, my_f1_score, rare_f1_score, out_labels, logits

    def calculate_communication(self, batch, comm_stats, comm_stats_with_server, is_mmlp, train_model=-1):
        if not is_mmlp:
            nodes = []
            # get k_hop from run_config num hidden
            if "gat" in self.model.model_name:
                k_hop = len(self.model.convs)
            else:
                k_hop = len(self.model.linear_layers_list)
            
            for i in range(batch.batch_size):
                nodes.append((i, -1, 0))
                
            while nodes:
                node = nodes.pop(0)
                node_id = node[0]
                parent = node[1]
                hop = node[2]
                neighbors = batch.edge_index[0][batch.edge_index[1] == node_id]
                
                if hop == 0:
                    comm_stats_with_server[batch.n_id[node_id].item()]["layers"][:] += 1
                    comm_stats_with_server[batch.n_id[node_id].item()]["layers"][:] += 1
                    
                if parent != -1:
                    comm_stats[batch.n_id[node_id].item()]["embeddings"][1: k_hop - hop + 1] += 1
                    comm_stats[batch.n_id[parent].item()]["embeddings"][1: k_hop - hop + 1] += 1
                        
                    if batch.n_id[node_id].item() not in comm_stats[batch.n_id[parent].item()]["nei_embeddings"][0]:
                        comm_stats[batch.n_id[node_id].item()]["embeddings"][0] += 1
                        comm_stats[batch.n_id[parent].item()]["embeddings"][0] += 1
                            
                        comm_stats[batch.n_id[parent].item()]["nei_embeddings"][0].add(batch.n_id[node_id].item())
    
                for nei in neighbors:
                    if hop < k_hop: 
                        if nei != parent:
                            comm_stats[batch.n_id[node_id].item()]["layers"][:k_hop - (hop + 1)] += 1
                            comm_stats[batch.n_id[nei].item()]["layers"][:k_hop - (hop + 1)] += 1
                        
                        comm_stats[batch.n_id[node_id].item()]["gradients"][:k_hop - (hop + 1)] += 1
                        comm_stats[batch.n_id[nei].item()]["gradients"][:k_hop - (hop + 1)] += 1
                        
                    if hop + 1 <= k_hop:
                        nodes.append((nei, node_id, hop + 1))
        else:
            nodes = []
            k_hop = train_model
            
            for i in range(batch.batch_size):
                nodes.append((i, -1, 0))
                
            while nodes:
                node = nodes.pop(0)
                node_id = node[0]
                parent = node[1]
                hop = node[2]
                neighbors = batch.edge_index[0][batch.edge_index[1] == node_id]
                
                if hop == 0:
                    comm_stats_with_server[batch.n_id[node_id].item()]["layers"][train_model] += 1
                    comm_stats_with_server[batch.n_id[node_id].item()]["layers"][train_model] += 1
                        
                        
    def stopping_criteria(self, epoch, best_epoch, early_stopping_patience):
        return epoch - best_epoch > early_stopping_patience

def lk_f1_score(preds, labels):
    return (
        f1_score(labels.cpu(), preds.detach().cpu(), average="micro"),
        f1_score(labels.cpu(), preds.detach().cpu(), average="macro"),
        f1_score(labels.cpu(), preds.detach().cpu(), average="weighted"),
    )

def rare_class_f1(output, labels):
    # identify the rare class
    ind = [torch.where(labels == 0)[0], torch.where(labels == 1)[0]]
    rare_class = int(len(ind[0]) > len(ind[1]))

    preds = F.softmax(output, dim=1).max(1)

    ap_score = average_precision_score(
        labels.cpu() if rare_class == 1 else 1 - labels.cpu(), preds[0].detach().cpu()
    )

    preds = preds[1].type_as(labels)

    TP = torch.sum(preds[ind[rare_class]] == rare_class).item()
    T = len(ind[rare_class])
    P = torch.sum(preds == rare_class).item()

    if P == 0:
        return (0, 0, 0, 0)

    precision = TP / P
    recall = TP / T
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1, precision, recall, ap_score