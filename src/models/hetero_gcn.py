from models.general_models import *
from scipy import sparse
from torch_scatter import scatter 
import numpy as np


class GCNSep(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden, dropout=0.0, model_name="gcn", feed_hidden_layer=False
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
                              
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs
    ):
        if not "edge_index" in kwargs:
            raise ValueError("edge index not found in kwargs")
        
        edge_index = kwargs["edge_index"]

        for i in range(self.num_hidden-1):
            x_1 = x.clone()             
            adj_row = edge_index[0]
            adj_col = edge_index[1]
            nei_x = scatter(x[adj_row], adj_col, dim=-2, reduce="mean", dim_size=x.size()[0])
            x = torch.cat((x, nei_x), dim = 1)
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            if i>=1:
                x = x + x_1

        
        adj_row = edge_index[0]
        adj_col = edge_index[1]
        
        nei_x = scatter(x[adj_row], adj_col, dim=-2, reduce="mean", dim_size=x.size()[0])
        x = torch.cat((x, nei_x), dim = 1)
        x = self.linear_layers_list[self.num_hidden-1](x)
        
        if labels is None:
            return x
        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss
    
    def get_layers_size(self):
        param_size = []
        param_count = 0
        for param in self.parameters():
            param_size.append((param.nelement() * param.element_size())/1024 ** 2)
            param_count += 1
            
        buffer_size = []
        buffer_count = 0
        for buffer in self.buffers():
            buffer_size.append((buffer.nelement() * buffer.element_size())/1024 ** 2)
            buffer_count += 1

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
        
if __name__ == "__main__":
    model = GCNSep(10, 8, 2)
    print(model)
    print(model.forward(torch.rand(10, 10), torch.rand(10, 10)))