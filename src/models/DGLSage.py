import dgl 
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class DGLMaxPoolAggregator(nn.Module):
    def __init__(
        self, input_size, output_size, device=None
    ):
        super(DGLMaxPoolAggregator, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.W1 = nn.Linear(input_size, output_size, bias=False).to(device)
        
                # norm_h = self.W1(h)
                # g.ndata['h'] = norm_h
                # g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.max('m', 'h_N'))
                # h_N = g.ndata['h_N']
                # return torch.cat([h, h_N], dim=1)
    def forward(self, g, feat, **kwargs):
        with g.local_scope():
            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
                feat_src = feat_dst = feat
            
            norm_h = self.W1(feat_src)
            g.srcdata['h'] = norm_h
            msg_fn = fn.copy_u('h', 'm')

            g.update_all(message_func=msg_fn, reduce_func=fn.max('m', 'h_N'))
            h_N = g.dstdata['h_N']
            
            return torch.cat([feat_dst, h_N], dim=1)

class DGLSage(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden, device=None, pool_size=512, dropout=0.0, model_name='DGLGraphSage'
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers_list = nn.ModuleList()
        self.aggregators_list = nn.ModuleList()
        self.model_name = model_name
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_hidden = num_hidden
        
        aggregator = DGLMaxPoolAggregator(input_size, pool_size, device=device)
        self.aggregators_list.append(aggregator)
        layer = nn.Linear(input_size + pool_size, hidden_size, bias=False).to(device)
        self.linear_layers_list.append(layer)
        
        for _ in range(1, self.num_hidden-1):
            aggregator = DGLMaxPoolAggregator(hidden_size, pool_size, device=device)
            self.aggregators_list.append(aggregator)
            layer = nn.Linear(hidden_size + pool_size, hidden_size, bias=False).to(device)
            self.linear_layers_list.append(layer)
            
        aggregator = DGLMaxPoolAggregator(hidden_size, pool_size, device=device)
        self.aggregators_list.append(aggregator)
        layer = nn.Linear(hidden_size + pool_size, output_size, bias=False).to(device)
        self.linear_layers_list.append(layer)
        
    def forward(self, mfgs, x: torch.Tensor, **kwargs):
        for i in range(self.num_hidden-1):
            x_dst = x[:mfgs[i].num_dst_nodes()]
            x = self.aggregators_list[i](mfgs[i], (x, x_dst))
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x_dst = x[:mfgs[-1].num_dst_nodes()]  
        x = self.aggregators_list[-1](mfgs[-1], (x, x_dst))
        x = self.linear_layers_list[-1](x)
        
        return x
    