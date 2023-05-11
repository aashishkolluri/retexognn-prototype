from models.general_models import *
from models.mlp import MLP
from torch_scatter import scatter
import numpy as np

class MLPPoolSep(MLP):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        pool_size=512,
        dropout=0.0,
        model_name="mlp_pool",
        num_hidden=2,
        rng=np.random
    ):
        super(MLPPoolSep, self).__init__(input_size+pool_size, hidden_size, output_size, dropout, num_hidden, model_name)

        self.pool_linear_layer = nn.Linear(input_size, pool_size, bias=False)
        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **kwargs):        
        if not "edge_index" in kwargs:
            raise ValueError("edge index not found in kwargs")

        x_c = x.clone()
        edge_index = kwargs["edge_index"]
                
        norm_m = self.pool_linear_layer(x)
        
        adj_row = edge_index[0]
        adj_col = edge_index[1]
        
        adj_row = adj_row.to(norm_m.device)
        adj_col = adj_col.to(norm_m.device)
        
        max_pooled_features = scatter(norm_m[adj_row], adj_col, dim=-2, reduce="max", dim_size=x.size()[0])
        
        # x_c = x.clone()
        x = torch.cat((x, max_pooled_features), dim=1)
        
        for i in range(self.num_hidden-1):
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear_layers_list[self.num_hidden-1](x)

        if not ("last_layer" in kwargs and kwargs["last_layer"]):
            x = x + x_c

        if labels is None:
            return x
        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss

class MultiMLPGCNSep(GeneralMultiMLPModel):
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
        model_name="mmlp_gcn",
    ):
        super(MultiMLPGCNSep, self).__init__(run_config, input_size, output_size, device, rng, eps, num_hidden, model_name)

        self.model_name = model_name+"_nl_"+str(self.nl)
        self.input_size = input_size
        self.output_size = output_size
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
        
        for it in range(1, run_config.nl+1):
            model = MLPPoolSep(
                        model_name=f"{model_name}_mlpsage_{self.nl}_{it}",
                        input_size=input_size,
                        hidden_size=self.hidden_size,
                        output_size=output_size,
                        dropout=self.dropout_val,
                        num_hidden=num_hidden
                    )
            self.model_list.append(model)
            input_size = output_size

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, num_model = None, **kwargs):
        if not "edge_index" in kwargs:
            raise ValueError("edge index not in kwargs")
        
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
            outputs = self.model_list[it](torch.cat(ft_nl, dim=1), labels, **{"edge_index": kwargs["edge_index"]})
            if labels is not None:
                outputs = outputs[0]
                loss = outputs[1]
            ft_nl = [outputs.detach()]
        
        outputs = self.model_list[-1](torch.cat(ft_nl, dim=1), labels, **{"edge_index": kwargs["edge_index"], "last_layer": True})

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

    model = MultiMLPGCNSep(RunConfig(), 10, 2, torch.rand(10, 10), "cpu", np.random)
    print(model)
    print(model.forward(torch.rand(10, 10)))