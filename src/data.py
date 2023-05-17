import torch
import torch_geometric
from torch_geometric.datasets import Planetoid, FacebookPagePage, LastFMAsia, KarateClub
import numpy as np
from utils import Dataset
import os
import time

from globals import MyGlobals

from get_inductive_data import get_inductive_data

INF = np.iinfo(np.int64).max

class LoadData:
    def __init__(
        self,
        dataset: Dataset,
        load_dir="planetoid",
        rng=None,
        rng_seed=None,
        test_dataset=None,
        split_num_for_dataset = 0,
        inductive = False
    ):
        self.dataset = dataset
        self.inductive = inductive
        self.load_dir = os.path.join(MyGlobals.DATADIR, load_dir)
        if not dataset.value in ["cora", "citeseer", "pubmed"] and load_dir == "planetoid":
            self.load_dir = os.path.join(MyGlobals.DATADIR, dataset.value)

        self.rng = rng
        self.rng_seed = rng_seed

        self.test_dataset = test_dataset
        self.features = None # N \times F matrix
        self.labels = None # N labels
        self.num_classes = None

        self.train_features = None
        self.train_labels = None

        self.val_features = None
        self.val_labels = None

        self.test_features = None
        self.test_labels = None

        self.edge_index = None
        self.data = None 
        self.split_num_for_dataset = split_num_for_dataset
        self._load_data() # fills in the values for above fields.

    def is_inductive(self):
        if self.inductive:
            return True
        
        return False 
    
    def val_on_new_graph(self):
        if self.inductive:
            return True
        return False

    def _get_masks_fb_page(self, dataset, te_tr_split=0.2, val_tr_split=0.2):
        nnodes = len(dataset[0].x)
        # get train mask, test mask and validation mask
        # get an 80-20 split for train-test
        # get an 80-20 split for train-val in train

        nodes = np.array(range(nnodes))
        train_mask = np.array([False] * nnodes)
        test_mask = np.array([False] * nnodes)
        val_mask = np.array([False] * nnodes)

        test_ind = self.rng.choice(nodes, int(te_tr_split * nnodes), replace=False)
        test_mask[np.array(test_ind)] = True
        rem_ind = []
        for ind in range(nnodes):
            if not ind in test_ind:
                rem_ind.append(ind)
        val_ind = self.rng.choice(rem_ind, int(val_tr_split * (len(rem_ind))), replace=False)
        val_mask[np.array(val_ind)] = True
        train_mask[~(test_mask | val_mask)] = True
        
        train_mask = torch.Tensor(train_mask)
        train_mask = train_mask.to(torch.bool)
        
        val_mask = torch.Tensor(val_mask)
        val_mask = val_mask.to(torch.bool)
        
        test_mask = torch.Tensor(test_mask)
        test_mask = test_mask.to(torch.bool)
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        # Testing
        assert ~(train_mask & test_mask).all()
        assert ~(val_mask & test_mask).all()
        assert ~(train_mask & val_mask).all()

        return train_mask, val_mask, test_mask

    def _load_data(self):
        print(
                "Load dir {}; dataset name {}".format(self.load_dir, self.dataset.name)
            )
        start_time = time.time()
        # This dataset is used for transfer learning so uses test_dataset for pred

        if self.dataset == Dataset.facebook_page:
            dataset = FacebookPagePage(self.load_dir)
            features = dataset[0].x
            labels = dataset[0].y
            # get number of nodes
            train_mask, val_mask, test_mask = self._get_masks_fb_page(dataset, 0.9, 0.5)
            edge_index = dataset[0].edge_index                   
        elif self.dataset == Dataset.LastFM:
            dataset = LastFMAsia(self.load_dir)
            features = dataset[0].x
            labels = dataset[0].y
            train_mask, val_mask, test_mask = self._get_masks_fb_page(dataset, 0.8, 0.5)
            edge_index = dataset[0].edge_index
        elif self.dataset == Dataset.KarateClub:
            dataset = [KarateClub(self.load_dir).data]
            features = dataset[0].x
            labels = dataset[0].y
            train_mask = dataset[0].train_mask
            val_mask = torch.full(dataset[0].train_mask.size(),False)
            test_mask = ~dataset[0].train_mask.clone()
            edge_index = dataset[0].edge_index
        
        elif self.dataset in [Dataset.Cora, Dataset.CiteSeer, Dataset.PubMed]:
            # use to load consistent dataset
            # dataset = Planetoid(root=self.load_dir, name=self.dataset.name, split='public')
            num_split = {
                'cora': [40, 280, 2148],
                'citeseer': [56, 336, 2655],
                'pubmed': [329, 987, 17743],
            }
            
            dataset = Planetoid(root=self.load_dir, name=self.dataset.name, split='random', num_train_per_class=num_split[self.dataset.name.lower()][0], num_val=num_split[self.dataset.name.lower()][1], num_test=num_split[self.dataset.name.lower()][2])
            features = dataset[0].x
            labels = dataset[0].y
            train_mask = dataset[0].train_mask
            val_mask = dataset[0].val_mask
            test_mask = dataset[0].test_mask
            edge_index = dataset[0].edge_index
            
            features = dataset[0].x
            labels = dataset[0].y
            
            # over_writing the train_mask, val_mask, test_mask different from the original given by the dataset.
            # train_mask, val_mask, test_mask = self._get_masks_fb_page(dataset, 0.9, 0.5)
        elif self.dataset in [Dataset.WikiCooc, Dataset.Roman]:
            fn = os.path.join(f"./data/heterophilous",f"{self.dataset.value}.npz")
            dataset = np.load(fn)
            train_mask = torch.tensor(dataset['train_masks'][self.split_num_for_dataset], dtype=torch.bool)
            val_mask = torch.tensor(dataset['val_masks'][self.split_num_for_dataset], dtype=torch.bool)
            test_mask = torch.tensor(dataset['test_masks'][self.split_num_for_dataset], dtype=torch.bool)
            edge_index = torch.tensor(dataset['edges'], dtype=torch.int64).T
            features = torch.tensor(dataset['node_features'], dtype=torch.float32)
            labels = torch.tensor(dataset['node_labels'], dtype=torch.int64)
            dataset = [torch_geometric.data.Data(x=features, edge_index=edge_index, y=labels)]
            dataset[0].train_mask = train_mask
            dataset[0].val_mask = val_mask
            dataset[0].test_mask = test_mask
        else:
            print(f"Dataset Loading undefined for {self.dataset.value}")
            exit()
        
        data = dataset[0]
        
        if self.dataset in [Dataset.WikiCooc, Dataset.Roman]:
            orig_edge_index = data.edge_index.clone().detach()
            
            new_edge_index_0 = torch.cat((data.edge_index[0], orig_edge_index[1]), dim=0)
            new_edge_index_1 = torch.cat((data.edge_index[1], orig_edge_index[0]), dim=0)
            
            data.edge_index = torch.stack([new_edge_index_0, new_edge_index_1]) 
            edge_index = data.edge_index
        try:
            data.n_id = torch.arange(dataset[0].num_nodes) # is not defined for inductive 
        except:
            data.n_id = torch.arange(dataset[0].num_nodes())
                    
        self.train_data = data
        self.val_data = data
        self.test_data = data
        self.features = features.clone()
        # print("len(features) {}".format(len(features)))

        self.train_features = self.features
        self.val_features = self.features
        self.test_features = self.features
        self.labels = labels.clone()
        self.train_labels = self.labels
        self.val_labels = self.labels
        self.test_labels = self.labels

        self.edge_index = edge_index
        self.train_edge_index = edge_index
        self.val_edge_index = edge_index
        self.test_edge_index = edge_index
        if self.inductive:
            self.train_data, self.val_data, self.test_data = get_inductive_data(dataset, self.rng)
            self.train_features = self.train_data.x
            self.val_features = self.val_data.x
            self.test_features = self.test_data.x
            self.train_labels = self.train_data.y
            self.val_labels = self.val_data.y
            self.test_labels = self.test_data.y
            self.train_edge_index = self.train_data.edge_index
            self.val_edge_index = self.val_data.edge_index
            self.test_edge_index = self.test_data.edge_index
            train_mask = self.train_data.train_mask
            val_mask = self.val_data.val_mask
            test_mask = self.test_data.test_mask

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        try:
            self.num_classes = len(set(self.labels.numpy()))
        except:
            self.num_classes = self.labels.numpy().shape[1]
            
        print(
            "{} {} {}".format(
                (self.train_mask == True).sum().item(),
                (self.val_mask == True).sum().item(),
                (self.test_mask == True).sum().item(),
            )
        )
        print(f"Data loading done: {time.time()-start_time}")
