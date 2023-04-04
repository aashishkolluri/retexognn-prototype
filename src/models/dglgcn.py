import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import dgl.function as fn


class DGLGCN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden, device=None, dropout=0.0, model_name="dglgcn", feed_hidden_layer=False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers_list = nn.ModuleList()
        self.linear_layers_list.append(nn.Linear(2*input_size, hidden_size, bias=False))
        
        for _ in range(1, self.num_hidden-1):
            self.linear_layers_list.append(nn.Linear(2*hidden_size, hidden_size, bias=False))
        self.linear_layers_list.append(nn.Linear(2*hidden_size, output_size, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
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
        
    def forward(self, mfgs, feat, **kwargs):
        for i in range(self.num_hidden-1):
            with mfgs[i].local_scope():
                feat_src = mfgs[i].srcdata["feat"]
                feat_dst = mfgs[i].dstdata["feat"]
                
                msg_fn = fn.copy_u('h', 'm')
                mfgs[i].srcdata['h'] = feat_src
                
                mfgs[i].update_all(message_func=msg_fn, reduce_func=fn.mean('m', 'h_N'))
                h_N = mfgs[i].dstdata['h_N']
                
                feat_src = torch.cat((feat_dst, h_N), dim=1)
                feat_src = self.linear_layers_list[i](feat_src)
                
                feat_src = self.relu(feat_src)
                feat_src = self.dropout(feat_src)
                mfgs[i + 1].srcdata["feat"] = feat_src
                mfgs[i + 1].dstdata["feat"] = feat_src[:mfgs[i + 1].num_dst_nodes()]
                
        with mfgs[-1].local_scope():
            feat_src = mfgs[-1].srcdata["feat"]
            feat_dst = mfgs[-1].dstdata["feat"]
            
            msg_fn = fn.copy_u('h', 'm')
            mfgs[-1].srcdata['h'] = feat_src
            mfgs[-1].update_all(message_func=msg_fn, reduce_func=fn.mean('m', 'h_N'))
            h_N = mfgs[-1].dstdata['h_N']
            
            feat_src = torch.cat((feat_dst, h_N), dim=1)
            feat_src = self.linear_layers_list[-1](feat_src)
            
        return feat_src
                
                
                
                