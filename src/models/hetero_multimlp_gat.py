import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import scipy.sparse as sp
from scipy import sparse
from models.general_models import *
from models.mlp import MLP
from models.hetero_gat import GraphAttentionLayer

class GATPoolSep(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, num_heads, lin_size, concat):
        super(GATPoolSep, self).__init__()
        
        self.dropout = dropout
        self.model_name = "gatpool"
        
        self.attentions = [GraphAttentionLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=concat) for _ in range(num_heads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hidden_l = nn.Linear(2*lin_size, lin_size)
        self.linear_layer = nn.Linear(lin_size, output_size, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if not 'edge_index' in kwargs:
            raise ValueError("edge_index not found in kwargs")
        edge_index = kwargs['edge_index']

        x_c = x.clone()
        x = F.dropout(x, self.dropout, training=self.training)
    
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hidden_l(x)
        x = self.linear_layer(x)

        x = x + x_c

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss 
    
    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.linear_layer.weight.device # hacky way to get the device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)    
        
class MultiMLPGATSep(GeneralMultiMLPModel):
    def __init__(
        self,
        run_config,
        input_size,
        output_size,
        device,
        rng,
        eps=0.0,
        feed_hidden_layer=False,
        num_hidden=2,
        model_name="mmlp_gat"
    ):
        super(MultiMLPGATSep, self).__init__(run_config, input_size, output_size, device, rng, eps, num_hidden, model_name)
        
        self.nl = run_config.nl
        self.model_name = model_name + "_nl_" + str(self.nl) + "_num_heads_" + str(run_config.attn_heads)
        self.feed_hidden_layer = feed_hidden_layer
        
        model = MLP(
            model_name=f"{model_name}_mlp_{self.nl}_{0}",
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=output_size,
            dropout=self.dropout_val,
            num_hidden=num_hidden,
        )
        self.model_list.append(model)

        input_size = output_size
        if self.feed_hidden_layer:
            input_size = self.hidden_size

        for _ in range(1, self.nl):
            self.model_list.append(GATPoolSep(
                input_size=input_size,
                hidden_size=run_config.hidden_size // run_config.attn_heads,
                output_size=output_size,
                dropout=self.dropout_val,
                alpha=0.2,
                num_heads=run_config.attn_heads,
                lin_size=self.hidden_size,
                concat=True,
            ))
            input_size = output_size 

        self.model_list.append(GATPoolSep(
            input_size=input_size,
            hidden_size=run_config.hidden_size,
            output_size=output_size,
            dropout=self.dropout_val,
            alpha=0.2,
            num_heads=1,
            lin_size=self.hidden_size,
            concat=False
        ))
        

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, num_model = None, **kwargs):
        if not "edge_index" in kwargs:
            raise ValueError("edge_index not found in kwargs")
        if not num_model is None:
            return self.model_list[num_model].forward(x, labels, **{"edge_index": kwargs["edge_index"]})

        outputs = self.model_list[0](x, labels)
        if labels is not None:
            loss = outputs[1]
            outputs = outputs[0]
        
        ft_nl = [outputs.detach()]
        
        if self.feed_hidden_layer:
            ft_nl = [self.model_list[0].hidden_features]
        
        for it in range(1, self.nl):
            outputs = self.model_list[it](torch.cat(ft_nl, 1), labels, **{"edge_index": kwargs["edge_index"]})
            if labels is not None:
                loss = outputs[1]
                outputs = outputs[0]
            ft_nl = [outputs.detach()]
        outputs = self.model_list[-1](torch.cat(ft_nl, 1), labels, **{"edge_index": kwargs["edge_index"], "last_layer": True})
        if labels is None:
            return outputs

        return outputs, loss
    
    def get_models_size(self):
        models_size = []
        
        for model in self.model_list:
            model_size = 0
            for param in model.parameters():
                model_size += (param.nelement() * param.element_size())/1024 ** 2
                
            models_size.append(model_size)
        
        return models_size
    
    def get_embeddings_size(self):
        embeddings_size = []
        embeddings_size.append((self.input_size * 4) / 1024 ** 2)
        
        for _ in range(self.nl):
            embeddings_size.append((self.output_size * 4) / 1024 ** 2)
            
        return embeddings_size

if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class RunConfig:  # Later overwritten in the main function. Only declaration and default initailization here.
        learning_rate: float = 0.01
        num_epochs: int = 200
        weight_decay: float = 5e-4
        num_warmup_steps: int = 0
        save_each_epoch: bool = False
        save_epoch: int = 50
        output_dir: str = "."
        hidden_size: int = 10
        num_hidden: int = 2
        dropout: float = 0.0
        nl: int = 2
        sample_list: list = None

    model = MultiMLPGATSep(RunConfig(), 10, 2, torch.rand(10, 10), "cpu", np.random)
    print(model)
    print(model.forward(torch.rand(10, 10)))       
        
        
        
               