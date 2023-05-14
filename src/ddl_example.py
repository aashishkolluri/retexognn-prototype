import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
import torch.nn as nn 
import torch.optim as optim 
import sklearn.metrics
import torch.nn.parallel 
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
from models.dglgcn import DGLGCN
from ogb.nodeproppred import DglNodePropPredDataset
from torch.optim import SGD
from tqdm import tqdm

def run(proc_id, devices):
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
    )
    
    if torch.cuda.device_count() < 1:
        device = torch.device("cpu")
        dist.init_process_group(
            backend="gloo",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device("cuda:" + str(dev_id))
        dist.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id
        )
        
    dataset = DglNodePropPredDataset("ogbn-products")
    
    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata["label"] = node_labels[:, 0]
    
    node_features = graph.ndata["feat"]
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()
    
    model = DGLGCN(input_size=num_features,
                hidden_size=128,
                output_size=num_classes,
                dropout=0.0,
                num_hidden=2).to(device)

    idx_split = dataset.get_idx_split()
    train_nids = idx_split["train"]
    valid_nids = idx_split["valid"]
    test_nids = idx_split["test"] 
    
    sampler = dgl.dataloading.NeighborSampler([-1, -1, -1])
    train_dataloader = dgl.dataloading.DataLoader(
        graph,
        train_nids,
        sampler,
        device=device,
        use_ddp=True,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )   
    
    val_dataloader = dgl.dataloading.DataLoader(
        graph,
        valid_nids,
        sampler,
        device=device,
        use_ddp=False,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    
    opt = SGD(model.parameters(),
              lr=0.01,
              momentum=0.9,
              weight_decay=5e-4)
    
    best_accuracy = 0
    best_model_path = "./model.pt"
    
    for epoch in range(2000):
        model.train()
        
        with tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                
                predictions = model(mfgs, inputs)
                
                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )
                
                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )

        model.eval()
        
        if proc_id == 0:
            predictions = []
            labels = []
            with tqdm(val_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    input = mfgs[0].srcdata["feat"]
                    labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    predictions.append(
                        model(mfgs, inputs).argmax(1).cpu.numpy()
                    )
                    predictions = np.concatenate(predictions)
                    labels = np.concatenate(labels)
                    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                    print("Epoch {} Validation Accuracy {}".format(epoch, accuracy))
                    
                    if best_accuracy < accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    num_gpus = 2
    import torch.multiprocessing as mp
    mp.spawn(run, args=(list(range(num_gpus)),), nprocs=num_gpus)
    
