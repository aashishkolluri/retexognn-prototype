import torch
import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

def get_inductive_data(dataset, rng):
    data = dataset[0]
    num_nodes = data.x.shape[0]
    num_train = int(num_nodes * 0.50)
    num_val = int(num_nodes * 0.1)
    num_test = num_nodes - num_train - num_val
    val_idx = list(rng.choice(num_nodes, num_val, replace=False))
    rem_idx = [x for x in range(num_nodes) if x not in val_idx]
    test_idx = list(rng.choice(rem_idx, num_test, replace=False))
    train_idx = [x for x in rem_idx if x not in test_idx]

    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0

    train_data = Data(x=data.x[train_idx], y=data.y[train_idx])
    train_data.edge_index = subgraph(train_idx, data.edge_index, relabel_nodes=True,)[0]
    train_data.train_mask = torch.zeros(num_train, dtype=torch.bool)
    act_train_nodes = rng.choice(num_train, int(num_train * 0.1), replace=False)
    train_data.train_mask[act_train_nodes] = True

    val_data = Data(x=data.x[train_idx + val_idx], y=data.y[train_idx + val_idx])
    val_data.edge_index = subgraph(train_idx + val_idx, data.edge_index, relabel_nodes=True)[0]
    val_data.val_mask = torch.full((num_train + num_val,), False, dtype=torch.bool)
    val_data.val_mask[num_train:] = True

    test_data = Data(x=data.x, y=data.y)
    test_data.edge_index = data.edge_index
    test_data.test_mask = torch.full((num_nodes,), False, dtype=torch.bool)
    test_data.test_mask[test_idx] = True

    return train_data, val_data, test_data


if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/cora', name='cora')
    train_data, val_data, test_data = get_inductive_data(dataset, np.random.RandomState(0))