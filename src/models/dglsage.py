import dgl 
import os
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
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W1.weight, gain=gain)
                
    def forward(self, g, feat, **kwargs):
        with g.local_scope():
            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
                feat_src = feat_dst = feat
            
            msg_fn = fn.copy_u('h', 'm')
            g.srcdata['h'] = self.W1(feat_src)

            g.update_all(message_func=msg_fn, reduce_func=fn.max('m', 'h_N'))
            h_N = g.dstdata['h_N']
            
            return torch.cat((feat_dst, h_N), dim=1)

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
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        for layer in self.linear_layers_list:
            nn.init.xavier_uniform_(layer.weight, gain=gain)

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.linear_layers_list[0].weight.device # hacky way to get the device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)
   
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