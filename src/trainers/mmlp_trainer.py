from trainers.general_trainer import * # trainers.general_trainer

import json
import sys
sys.path.append("..")
from data import LoadData
from models.multimlp_gcn import MultiMLPGCN
from models.multimlp_sage import MultiMLPSAGE
from models.multimlp_gat import MultiMLPGAT
from models.hetero_multimlp_gcn import MultiMLPGCNSep
from models.hetero_multimlp_gat import MultiMLPGATSep
from models.hetero_multimlp_sage import MultiMLPSAGESep
from torch_geometric.loader import NeighborLoader
from scipy import sparse
import scipy.sparse as sp
import random
import math

def train_mmlp_like_models(
    run_config,
    dataset: utils.Dataset,
    device,
    arch,
    print_graphs=False,
    dp=False,
    seeds=[1],
    test_dataset=None,
    feed_hidden_layer=False,
    sample_neighbors=False,
    ):
    
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

        data_loader = LoadData(dataset, rng=rng, rng_seed=seeds[i], test_dataset=test_dataset, split_num_for_dataset = i%10, inductive = run_config.inductive)
        
        data_loader.train_data = data_loader.train_data.to(device, 'x', 'y')
        data_loader.train_edge_index = data_loader.train_edge_index.to(device)
        
        num_classes = data_loader.num_classes
        train_features = data_loader.train_features
        train_labels = data_loader.train_labels
        num_nodes = len(train_labels)
        
        if run_config.diff_nei:
            neighbors = []
            for node in range(num_nodes):
                neighbors.append(data_loader.train_edge_index[1][data_loader.train_edge_index[0] == node].tolist())
            
            sampled_neighbors = [[], []]
            frac = run_config.frac
            
            for node in range(len(neighbors)):
                sampled_nei = random.sample(neighbors[node], math.ceil(len(neighbors[node]) * frac))
                for nei in sampled_nei:
                    sampled_neighbors[0].append(node)
                    sampled_neighbors[1].append(nei)
                    
                    sampled_neighbors[0].append(nei)
                    sampled_neighbors[1].append(node)
            
            sampled_neighbors = torch.Tensor(sampled_neighbors)
            sampled_neighbors = sampled_neighbors.to(torch.int64)
            
            data_loader.train_data.edge_index = sampled_neighbors
            data_loader.val_data.edge_index = sampled_neighbors
        
        val_features = data_loader.val_features
        val_labels = data_loader.val_labels
        
    
        num_train_nodes = (data_loader.train_mask == True).sum().item()
        
        batch_size = run_config.batch_size if run_config.batch_size else num_train_nodes
        num_neighbors = [25] if sample_neighbors else [-1]       

        train_loader = NeighborLoader(
            data_loader.train_data,
            num_neighbors=[0],
            batch_size=batch_size,
            input_nodes=data_loader.train_mask,
            **{'shuffle': True}
        )
        
        data_loader.val_data = data_loader.val_data.to(device, 'x', 'y')
        val_loader = NeighborLoader(
            data_loader.val_data,
            num_neighbors=[0],
            batch_size=batch_size,
            input_nodes=data_loader.val_mask,
            **{'shuffle': True}
        )
        
        if num_classes > 2:
            is_rare = False
            
        if data_loader.test_dataset:
            test_dataset = data_loader.test_dataset
        
        if run_config.hetero:
            mmlp = {utils.Architecture.MMLPGCN: MultiMLPGCNSep, 
                    utils.Architecture.MMLPSAGE: MultiMLPSAGESep,
                    utils.Architecture.MMLPGAT: MultiMLPGATSep}[arch](
                                                                    run_config=run_config,
                                                                    input_size=train_features.size(1),
                                                                    output_size=num_classes,
                                                                    num_hidden=run_config.num_hidden,
                                                                    device=device,
                                                                    rng=rng,
                                                                    feed_hidden_layer=feed_hidden_layer
                                                                    )
        else:
            mmlp = {utils.Architecture.MMLPGCN: MultiMLPGCN, 
                    utils.Architecture.MMLPSAGE: MultiMLPSAGE,
                    utils.Architecture.MMLPGAT: MultiMLPGAT}[arch](
                                                                    run_config=run_config,
                                                                    input_size=train_features.size(1),
                                                                    output_size=num_classes,
                                                                    num_hidden=run_config.num_hidden,
                                                                    device=device,
                                                                    rng=rng,
                                                                    feed_hidden_layer=feed_hidden_layer
                                                                    )

        print(
            "Input features {}, num_classes {}".format(
                train_features.size(1), num_classes
            )
        )
                
        model = mmlp.model_list[0]
        kwargs = {}

        if run_config.parse_communication_results:
            utils.parse_communication_mmlp(mmlp, dataset)
            return
                                
        if run_config.calculate_communication:        
            comm_stats = {}
            comm_stats_with_server = {}
            
            for x in range(num_nodes):
                layers_server = np.zeros(len(mmlp.model_list))
                embeddings = np.zeros(len(mmlp.model_list) + 1)
            
                comm_stats[x] = {
                    "embeddings": embeddings
                }
                
                comm_stats_with_server[x] = {
                    "layers": layers_server,
                }
            
            kwargs["comm_stats"] = comm_stats
            kwargs["comm_stats_with_server"] = comm_stats_with_server
            kwargs["is_mmlp"] = True
            kwargs["train_model"] = 0
            
        trainer = Trainer(model, rng, seed=seeds[i])
        original_lr = run_config.learning_rate
        run_config.learning_rate = 0.1
        
        _, _, best_epoch = trainer.train(
            dataset,
            train_loader,
            val_loader,
            device,
            run_config,
            test_dataset=test_dataset,
            kwargs=kwargs,
            feed_hidden_layer=feed_hidden_layer,
            sample_neighbors=sample_neighbors,
            store_result=False,
        )
        
        run_config.learning_rate = original_lr
        
        best_epochs.append(best_epoch)

        eval_mask = torch.zeros(train_features.size(0), dtype=torch.bool)
        eval_mask[:] = True
        
        eval_loader = NeighborLoader(
            data_loader.train_data,
            num_neighbors=num_neighbors,
            batch_size=eval_mask.sum().item(),
            input_nodes=eval_mask, 
        )
        
        logits = None
        labels = None
        for batch in eval_loader:
            model.eval()
            
            labels = batch.y
            kwargs["edge_index"] = batch.edge_index.to(batch.x.device)
            with torch.no_grad():
                logits = model(batch.x, **kwargs)

        del eval_loader
        
        features1 = logits.detach()
        labels = labels.detach()
        
        if feed_hidden_layer:
            features1 = model.hidden_features
        
        features1_val = features1
        labels_val = labels
        if data_loader.val_on_new_graph():
            # need validation features for inductive setting
            eval_mask = torch.zeros(val_features.size(0), dtype=torch.bool)
            eval_mask[:] = True
            eval_loader = NeighborLoader(
                data_loader.val_data,
                num_neighbors=num_neighbors,
                batch_size=eval_mask.sum().item(),
                input_nodes=eval_mask,
            )

            val_logits = None
            for batch in eval_loader:
                model.eval()
                
                labels_val = batch.y
                kwargs["edge_index"] = batch.edge_index.to(batch.x.device)
                with torch.no_grad():
                    val_logits = model(batch.x, **kwargs)

            features1_val = val_logits.detach()
            labels_val = labels_val.detach()

            del eval_loader
            
        ft_nl = [features1]
        ft_nl_val = [features1_val]

        del train_loader
        del val_loader

        for it in range(run_config.nl):
            if run_config.diff_nei:
                sampled_neighbors = [[], []]
                
                for node in range(len(neighbors)):
                    sampled_nei = random.sample(neighbors[node], math.ceil(len(neighbors[node]) * frac))
                    for nei in sampled_nei:
                        sampled_neighbors[0].append(node)
                        sampled_neighbors[1].append(nei)
                        
                        sampled_neighbors[0].append(nei)
                        sampled_neighbors[1].append(node)
                
                sampled_neighbors = torch.Tensor(sampled_neighbors)
                sampled_neighbors = sampled_neighbors.to(torch.int64)
                
                data_loader.train_data.edge_index = sampled_neighbors
                data_loader.val_data.edge_index = sampled_neighbors
            
            if run_config.calculate_communication:
                for node in comm_stats:
                    kwargs["comm_stats_with_server"][node]["layers"][it] += 1
                
                    neighbors = data_loader.train_data.edge_index[1][data_loader.train_data.edge_index[0] == node]
                    for nei in neighbors:
                        kwargs["comm_stats"][node]["embeddings"][it + 1] += 1
                        kwargs["comm_stats"][nei.item()]["embeddings"][it + 1] += 1
                    
            model1 = mmlp.model_list[it + 1]
            kwargs["train_model"] = it + 1
            trainer1 = Trainer(model1, rng, seed=seeds[i])
            features = torch.cat(ft_nl, dim=1)
            features_val = features

            feat_data = data_loader.train_data.clone()
            feat_data.x = features
            feat_data.y = labels
            feat_data = feat_data.to(device, 'x', 'y')
            
            train_loader = NeighborLoader(
                feat_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=data_loader.train_mask,
                **{'shuffle': True}
            )
            
            if data_loader.val_on_new_graph():
                features_val = torch.cat(ft_nl_val, dim=1)
            
            feats_val_data = data_loader.val_data.clone()
            feats_val_data.x = features_val
            feats_val_data.y = labels_val

            feats_val_data = feats_val_data.to(device, 'x', 'y')
            val_loader = NeighborLoader(
                feats_val_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=data_loader.val_mask,
                **{'shuffle': True}
            )
            
            kwargs1 = kwargs.copy()
            if it + 1 == run_config.nl:
                kwargs1["last_layer"] = True        
            val_loss, val_acc, best_epoch = trainer1.train(
                dataset,
                train_loader,
                val_loader,
                device,
                run_config,
                test_dataset=test_dataset,
                kwargs=kwargs1,
                feed_hidden_layer=feed_hidden_layer,
                sample_neighbors=sample_neighbors,
                store_result=False,
            )
            # Record the best_epoch for each model that makes up the MMLP
            best_epochs.append(best_epoch)
            # print("features1.size {}".format(features1.size(1)))

            # getting the train logits after validation since we need them for communities in the next iteration
            # also computing the test_acc and scores -- for the case of not a different test set these are also
            # the final test_acc
            
            eval_mask = torch.zeros(train_features.size(0), dtype=torch.bool)
            eval_mask[:] = True
            
            eval_loader = NeighborLoader(
                feat_data,
                num_neighbors=num_neighbors,
                batch_size=eval_mask.sum().item(),
                input_nodes=eval_mask,                
            )
            
            for batch in eval_loader:
                model1.eval()
                
                kwargs1["edge_index"] = batch.edge_index.to(batch.x.device)
                with torch.no_grad():
                    logits = model1(batch.x, **kwargs1)

            del eval_loader
            
            features1 = logits.detach()
            ft_nl = [features1]


            if data_loader.val_on_new_graph():
                # getting validation dataset logits
                eval_mask = torch.zeros(val_features.size(0), dtype=torch.bool)
                eval_mask[:] = True
                eval_loader = NeighborLoader(
                    feats_val_data,
                    num_neighbors=num_neighbors,
                    batch_size=eval_mask.sum().item(),
                    input_nodes=eval_mask,
                )
                
                val1_logits = None
                for batch in eval_loader:
                    model1.eval()
                    
                    kwargs1["edge_index"] = batch.edge_index.to(batch.x.device)
                    with torch.no_grad():
                        val1_logits = model1(batch.x, **kwargs1)

                
                del eval_loader                
                
                features1_val = val1_logits.detach()
                ft_nl_val = [features1_val]

            del train_loader
            del val_loader
        if run_config.calculate_communication:
            comm_result = {}
            comm_result_with_server = {}
            
            for node in kwargs["comm_stats"]:
                comm_result[node] = {
                    "embeddings": kwargs["comm_stats"][node]["embeddings"].tolist(),
                }
                
                comm_result_with_server[node] = {
                    "layers": kwargs["comm_stats_with_server"][node]["layers"].tolist()
                }

            with open(f"comm_results/{dataset}_{mmlp.model_name}.json", "w") as f:
                json.dump(comm_result, f, indent = 4, sort_keys=True)
                
            with open(f"comm_results/{dataset}_{mmlp.model_name}_with_server.json", "w") as f:
                json.dump(comm_result_with_server, f, indent = 4, sort_keys=True)
                        
        dir_name = utils.get_folder_name(
                    run_config,
                    dataset,
                    mmlp.model_name,
                    seeds[i],
                    test_dataset=test_dataset,
                    feed_hidden_layer=feed_hidden_layer,
                    sample_neighbors=sample_neighbors,
                    inductive=run_config.inductive
                )
        
        best_output_dir = os.path.join(run_config.output_dir, dir_name)
        if best_output_dir:
            mmlp.save(best_output_dir)
            print("Best saved at {}".format(best_output_dir))

        data_loader.test_data = data_loader.test_data.to(device, 'x', 'y')
        test_loader = NeighborLoader(
            data_loader.test_data,
            num_neighbors=num_neighbors * run_config.nl,
            batch_size=data_loader.test_mask.sum().item(),
            input_nodes=data_loader.test_mask,
        )
        
        mmlp.prepare_for_fwd()
        total_loss = total_correct = total_examples = total_rare_f1_score = 0
        total_f1_score_0 = total_f1_score_1 = total_f1_score_2 = 0
        
        out_labels  = []
        logits = []
        
        for batch in test_loader:
            y = batch.y[:batch.batch_size]
            
            kwargs['edge_index'] = batch.edge_index.to(batch.x.device)
            with torch.no_grad():
                y_hat = mmlp(batch.x, **kwargs)[:batch.batch_size]
            
            logits.append(y_hat.cpu().detach().numpy())
            
            loss = nn.CrossEntropyLoss()(y_hat, y)

            total_loss += float(loss) * batch.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += batch.batch_size

            preds = F.softmax(y_hat, dim=1)
            predicted_labels = torch.max(preds, dim=1).indices[:]
            out_labels.append(predicted_labels[1].cpu().detach().numpy())            
             
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
                                
        all_outputs.append((out_labels, logits))
        test_accuracies.append(acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(loss)

        f1_scores.append(my_f1_score)
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