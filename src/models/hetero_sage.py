import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
import torch.nn.functional as F
from models.general_models import *


class MaxPoolAggregator(nn.Module):
    def __init__(
        self, input_size=None, output_size=None, device='cpu', num_sample=None, rng=np.random,
    ):
        super(MaxPoolAggregator, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.num_sample = num_sample
        self.W1 = nn.Linear(input_size, output_size, bias=False)
    
    def forward(self, x, **kwargs):
        if not "edge_index" in kwargs:
            raise ValueError("edge index not found in kwargs")
        
        edge_index = kwargs["edge_index"]
         
        norm_m = self.W1(x)

        adj_row = edge_index[0]
        adj_col = edge_index[1]
        
        adj_row = adj_row.to(norm_m.device)
        adj_col = adj_col.to(norm_m.device)  

        max_pooled_features = scatter(norm_m[adj_row], adj_col, dim=-2, reduce="max", dim_size=x.size()[0])
        return torch.cat((x, max_pooled_features), dim=1)

class GraphSAGESep(GeneralModel):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden, device=None, pool_size=512, dropout=0.0, model_name='GraphSAGE', feed_hidden_layer=False
    ):
        super(GraphSAGESep, self).__init__(input_size, hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers_list = nn.ModuleList()
        self.aggregators_list = nn.ModuleList()
        self.model_name = model_name
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_hidden = num_hidden

        aggregator = MaxPoolAggregator(input_size, pool_size, device=device)
        self.aggregators_list.append(aggregator)
        layer = nn.Linear(input_size + pool_size, hidden_size, bias=False)
        self.linear_layers_list.append(layer)

        for _ in range(1, self.num_hidden-1):
            aggregator = MaxPoolAggregator(hidden_size, pool_size, device=device)
            self.aggregators_list.append(aggregator)
            layer = nn.Linear(hidden_size + pool_size, hidden_size, bias=False)
            self.linear_layers_list.append(layer) 

        aggregator = MaxPoolAggregator(hidden_size, pool_size, device=device)
        self.aggregators_list.append(aggregator)
        layer = nn.Linear(hidden_size + pool_size, output_size, bias=False)
        self.linear_layers_list.append(layer)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if not "edge_index" in kwargs:
            raise ValueError("edge index not found in kwargs")

        for i in range(self.num_hidden-1):
            x_1 = x.clone()
            x = self.aggregators_list[i](x, **{"edge_index": kwargs["edge_index"]})
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            if i>=1:
                x = x_1 + x

        x = self.aggregators_list[-1](x, **{"edge_index": kwargs["edge_index"]})
        x = self.linear_layers_list[-1](x)

        if labels is None:
            return x
        
        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss
    
    def get_layers_size(self):
        param_size = []
        for param in self.linear_layers_list.parameters():
            param_size.append((param.nelement() * param.element_size())/1024 ** 2)
            
        for i, param in enumerate(self.aggregators_list.parameters()):
            param_size[i] += (param.nelement() * param.element_size())/1024 ** 2
        
        buffer_size = []
        for buffer in self.buffers():
            buffer_size.append((buffer.nelement() * buffer.element_size())/1024 ** 2)

        return param_size, buffer_size

    def get_embeddings_size(self):
        embeddings_size = []
        embeddings_size.append((self.input_size * 4) / 1024 ** 2)
        embeddings_size.append((self.hidden_size * 4) / 1024 ** 2)
        
        for _ in range(1, self.num_hidden - 1):
            embeddings_size.append((self.hidden_size * 4) / 1024 ** 2)
            
        embeddings_size.append((self.output_size * 4) / 1024 ** 2) 
        
        return embeddings_size
        
    def get_gradients_size(self):
        gradients_size = []
        gradients_size.append((self.input_size * 4) / 1024 ** 2)
        gradients_size.append((self.hidden_size * 4) / 1024 ** 2)
        
        for _ in range(1, self.num_hidden - 1):
            gradients_size.append((self.hidden_size * 4) / 1024 ** 2)
        
        return gradients_size  
      