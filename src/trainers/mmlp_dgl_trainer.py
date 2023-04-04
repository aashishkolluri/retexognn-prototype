import os
import sys
# sys.path.append("..")
from data import LoadData
from models.multimlp_dglsage import MultiMLPDGLSage
from models.multimlp_dglgcn import MultiMLPDGLGCN
from trainers.general_trainer import set_torch_seed, Trainer, TrainStats
from trainers.general_dgl_trainer import DGLTrainer, lk_f1_score
from torch.optim import SGD, Adam, lr_scheduler
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

def train_mmlp_dgl_like_models(
    run_config,
    dataset: utils.Dataset,
    device,
    arch,
    seeds=[1],
    sample_neighbors=False,
    feed_hidden_layer=False
):

    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    rare_f1_scores = []
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
        num_neighbors = [25,25] if sample_neighbors else [-1,-1]
        
        train_ids = (train_mask == True).nonzero(as_tuple=True)[0]
        val_ids = (val_mask == True).nonzero(as_tuple=True)[0]
        test_ids = (test_mask == True).nonzero(as_tuple=True)[0]
        
        sampler = dgl.dataloading.NeighborSampler(num_neighbors)
        
        train_loader = dgl.dataloading.DataLoader(
            data_loader.train_data,
            train_ids,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        val_loader = dgl.dataloading.DataLoader(
            data_loader.val_data,
            val_ids,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        mmlp = {utils.Architecture.MMLPDGLSAGE: MultiMLPDGLSage,
                utils.Architecture.MMLPDGLGCN: MultiMLPDGLGCN}[arch](
            run_config=run_config,
            input_size=features.size(1),
            output_size=num_classes,
            device=device,
            rng=rng,
            num_hidden=run_config.num_hidden,
            feed_hidden_layer=feed_hidden_layer
        )
        
        print(
            "Input features {}, num_classes {}".format(
                features.size(1), num_classes
            )
        )
        
        model = mmlp.model_list[0]
        kwargs = {}
        
        trainer = DGLTrainer(model, rng, seed=seeds[i])
        original_lr = run_config.learning_rate
        run_config.learning_rate = 0.1
        
        _, _, best_epoch = trainer.train(
            dataset,
            train_loader,
            val_loader,
            device,
            run_config,
            kwargs=kwargs,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
            store_result=False,
        )
        
        run_config.learning_rate = original_lr
        
        best_epochs.append(best_epoch)
        
        eval_mask = torch.zeros(features.size(0), dtype=torch.bool)
        eval_mask[:] = True
        eval_ids = (eval_mask == True).nonzero(as_tuple=True)[0]
        
        eval_loader = dgl.dataloading.DataLoader(
            data_loader.train_data,
            eval_ids,
            sampler,
            device=device,
            batch_size=eval_mask.sum().item(),
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
     
        logits = None
        labels = None 
        
        for i, (input_nodes, output_nodes, mfgs) in enumerate(eval_loader):
            model.eval()
            
            inputs = mfgs[-1].dstdata['feat']
            labels = mfgs[-1].dstdata['label']
            logits = model(mfgs, inputs, **kwargs)
            
        del eval_loader
        
        features1 = logits.detach()
        labels = labels.detach()
        
        if feed_hidden_layer:
            features1 = model.hidden_features
        
        features1_val = features1
        labels_val = labels
        
        ft_nl = [features1]
        ft_nl_val = [features1_val]
        
        del train_loader
        del val_loader
        
        for it in range(run_config.nl):
            model1 = mmlp.model_list[it + 1]
            trainer1 = DGLTrainer(model1, rng, seed=seeds[i])
            features = torch.cat(ft_nl, dim=1)
            features_val = features 
            
            feat_data = data_loader.train_data.clone()
            feat_data.ndata['feat'] = features.detach().cpu()
            feat_data.ndata['label'] = labels.detach().cpu()
           
            train_loader = dgl.dataloading.DataLoader(
                feat_data,
                train_ids,
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0
            )
            
            feat_val_data = data_loader.val_data.clone()
            feat_val_data.ndata['feat'] = features_val.detach().cpu()
            feat_val_data.ndata['label'] = labels_val.detach().cpu()
            
            val_loader = dgl.dataloading.DataLoader(
                feat_val_data,
                val_ids,
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0
            )
            
            kwargs1 = kwargs.copy()
            
            val_loss, val_acc, best_epoch = trainer1.train(
                dataset,
                train_loader,
                val_loader,
                device,
                run_config,
                kwargs=kwargs1,
                feed_hidden_layer=feed_hidden_layer,
                sample_neighbors=sample_neighbors,
                store_result=False,
            )
            
            best_epochs.append(best_epoch)
            
            eval_mask = torch.zeros(features.size(0), dtype=torch.bool)
            eval_mask[:] = True
            eval_ids = (eval_mask == True).nonzero(as_tuple=True)[0]
               
            eval_loader = dgl.dataloading.DataLoader(
                feat_data,
                eval_ids,
                sampler,
                device=device,
                batch_size=eval_mask.sum().item(),
                shuffle=False,
                drop_last=False,
                num_workers=0       
            )

            for i, (input_nodes, output_nodes, mfgs) in enumerate(eval_loader):
                model1.eval()
                
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']
                logits = model1(mfgs, inputs, **kwargs1)
            
            del eval_loader
            
            features1 = logits.detach()
            ft_nl = [features1]
            
        dir_name = utils.get_folder_name(
            run_config,
            dataset,
            mmlp.model_name,
            seeds[i],
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
            inductive=run_config.inductive
        )   
        
        best_output_dir = os.path.join(run_config.output_dir, dir_name)
        if best_output_dir:
            mmlp.save(best_output_dir)
            print("Best saved at {}".format(best_output_dir))
        
        test_mask_orig = test_mask.clone().detach()
        test_ids_orig = test_ids.clone().detach()
        
        test_mask[:] = True
        test_ids = (test_mask == True).nonzero(as_tuple=True)[0]
        
        test_loader = dgl.dataloading.DataLoader(
            data_loader.test_data,
            test_ids,
            sampler,
            device=device,
            batch_size=test_mask.sum().item(),
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        mmlp.prepare_for_fwd()
        
        total_loss = total_correct = total_examples = 0
        total_f1_score_0 = total_f1_score_1 = total_f1_score_2 = 0
        
        out_labels = []
        logits = []
        kwargs = {}
        
        for i, (input_nodes, output_nodes, mfgs) in enumerate(test_loader):
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            predictions = mmlp(mfgs, inputs, **kwargs)
            
            predictions = predictions[test_ids_orig]
            labels = labels[test_ids_orig]
            current_batch_size = labels.size()[0]
            
            logits.append(predictions.cpu().detach().numpy())
            loss = nn.CrossEntropyLoss()(predictions, labels)
            
            total_loss += float(loss) * current_batch_size
            total_correct += int((predictions.argmax(dim=-1) == labels).sum())
            total_examples += current_batch_size
            
            predicted_labels = predictions.argmax(dim=-1)
            out_labels.append(predicted_labels.cpu().detach().numpy())
            
            f1_score_0, f1_score_1, f1_score_2 = lk_f1_score(predicted_labels, labels)
            
            total_f1_score_0 += float(f1_score_0) * current_batch_size
            total_f1_score_1 += float(f1_score_1) * current_batch_size
            total_f1_score_2 += float(f1_score_2) * current_batch_size
            
        loss = total_loss / total_examples
        acc = total_correct / total_examples
        my_f1_score = (total_f1_score_0 / total_examples, total_f1_score_1 / total_examples, total_f1_score_2 / total_examples)
        
        all_outputs.append((out_labels, logits))
        test_accuracies.append(acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(loss)
        f1_scores.append(my_f1_score)
        
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
                        
            
        
        
            
        
        
        
        
        