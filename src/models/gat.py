import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter
from models.general_models import *


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha 
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    
    def forward(self, h, adj):
        adj_row = adj[0]
        adj_col = adj[1]
        # add self loops
        adj_row = torch.cat([adj_row, adj_col.unique()], dim=0)
        adj_col = torch.cat([adj_col, adj_col.unique()], dim=0)
        Wh = torch.mm(h, self.W)
        Wh1, Wh2 = self._prepare_attentional_mechanisms_input(Wh)
        # get only the src node scores
        Wh1 = Wh1.index_select(0, adj_row)
        Wh2 = Wh2.index_select(0, adj_col)
        e = self.leakyrelu(Wh2 + Wh1)
        alpha = softmax(e.T.view(-1), adj_col)
        Wh_j = Wh[adj_row] * alpha.view(-1, 1)
        h_prime = scatter(Wh_j, adj_col, dim=-2, dim_size=h.size(0), reduce='sum')
        
        if self.concat:
            return F.elu(h_prime) 
        else:
            return h_prime
        
    
    def _prepare_attentional_mechanisms_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        return Wh1, Wh2
    
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden, num_heads, dropout=0.0, alpha=0.2, feed_hidden_layer=False,):
        super(GAT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout 
        self.model_name = f"gat-num_heads_{num_heads}"
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        self.convs = torch.nn.ModuleList()
        attentions = torch.nn.ModuleList()
        for _ in range(num_heads):
            attentions.append(GraphAttentionLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True))
        self.convs.append(attentions)
        for _ in range(1, self.num_hidden-1):
            attentions = torch.nn.ModuleList()
            for _ in range(num_heads):
                attentions.append(GraphAttentionLayer(hidden_size * num_heads, hidden_size, dropout=dropout, alpha=alpha, concat=True))
            self.convs.append(attentions)
        out_att = GraphAttentionLayer(hidden_size * num_heads, output_size, dropout=dropout, alpha=alpha, concat=False)
        self.convs.append(out_att)


    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if not 'edge_index' in kwargs:
            raise ValueError("edge_index not found in kwargs")

        edge_index = kwargs['edge_index']
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.convs[0]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(1, self.num_hidden-1):
            x = torch.cat([att(x, edge_index) for att in self.convs[i]], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.convs[-1](x, edge_index))
        x = F.log_softmax(x, dim=1)

        if labels is None:
            return x

        loss  = nn.CrossEntropyLoss()(x, labels)
        return x, loss

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.convs[0][0].W.device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)
        
    def get_layers_size(self):
        param_sizes = []
        buffer_sizes = []
        
        for conv in self.convs:
            param_size = 0
            for param in conv.parameters():
                param_size += (param.nelement() * param.element_size())/1024 ** 2

            param_sizes.append(param_size)
        
        
        for conv in self.convs:
            buffer_size = 0
            for buffer in conv.buffers():
                buffer_size += (buffer.nelement() * buffer.element_size())/1024 ** 2
            
            buffer_sizes.append(buffer_size)
            
        return param_sizes, buffer_sizes       
    
    def get_embeddings_size(self):
        embeddings_size = []
        embeddings_size.append((self.input_size * 4) / 1024 ** 2)
        embeddings_size.append((self.hidden_size * self.num_heads * 4) / 1024 ** 2)
        
        for _ in range(1, self.num_hidden - 1):
            embeddings_size.append((self.hidden_size * self.num_heads * 4) / 1024 ** 2)
            
        embeddings_size.append((self.output_size * 4) / 1024 ** 2) 
        
        return embeddings_size         
    
    def get_gradients_size(self):
        gradients_size = []
        gradients_size.append((self.input_size * 4) / 1024 ** 2)
        gradients_size.append((self.hidden_size * self.num_heads * 4) / 1024 ** 2)
        
        for _ in range(1, self.num_hidden - 1):
            gradients_size.append((self.hidden_size * self.num_heads * 4) / 1024 ** 2)
        
        return gradients_size