from models.general_models import *
from models.mlp import MLP
from scipy import sparse
from torch_scatter import scatter 
import numpy as np

class MLPGCN(MLP):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, num_hidden=2, model_name="mlpgcn", num_sample=-1):
        super().__init__(2*input_size, hidden_size, output_size, dropout, num_hidden, model_name)

        self.num_sample = num_sample   
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if not "edge_index" in kwargs:
            raise ValueError("edge index not found in kwargs")
        
        edge_index = kwargs["edge_index"]
        adj_row = edge_index[0]
        adj_col = edge_index[1]
                   
        nei_m = scatter(x[adj_row], adj_col, dim=-2, reduce="mean", dim_size=x.size()[0])
        
        x = torch.cat((x, nei_m), dim=1)
        x = x.to(self.linear_layers_list[0].weight.device)
        for i in range(self.num_hidden-1):
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear_layers_list[self.num_hidden-1](x)

        if labels is None:
            return x
        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss

class MultiMLPGCN(GeneralMultiMLPModel):
    def __init__(
        self,
        run_config,
        input_size,
        output_size,
        device,
        rng,
        feed_hidden_layer=False,
        sample_neighbors=False,
        eps=0.0,
        num_hidden=2,
        model_name="mmlp_gcn",
    ):
        super().__init__(run_config, input_size, output_size, device, rng, eps, num_hidden, model_name)

        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name+"_nl_"+str(self.nl)
        self.feed_hidden_layer = feed_hidden_layer
        
        # Initialize the first model
        model = MLP(
            model_name=f"{model_name}_mlp_{self.nl}_{0}",
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=output_size,
            dropout=self.dropout_val,
            num_hidden=self.num_hidden,
        )
        
        self.model_list.append(model)
        input_size = output_size
        
        if self.feed_hidden_layer:
            input_size = self.hidden_size
        
        for it in range(1, run_config.nl + 1):
            print("MMLP-{} input features size {}".format(it, input_size))
            model = MLPGCN(
                model_name=f"{model_name}_mlpgcn_{self.nl}_{it}",
                input_size=input_size,
                hidden_size=self.hidden_size,
                output_size=output_size,
                dropout=run_config.dropout,
                num_hidden=num_hidden
            )
            
            self.model_list.append(model)
            if it == run_config.nl:
                input_size = output_size
                
            input_size = output_size
            # input_size += 2 * output_size # += if you want to consider history, 2 otw

    def normalize(self, adjacency):
        D_1 = sparse.diags(np.power(np.array(adjacency.sum(1)), -1).flatten())
        adj_hat = D_1.dot(adjacency).tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((adj_hat.row, adj_hat.col)).astype(np.int64)
        )
        values = torch.from_numpy(adj_hat.data)
        return torch.sparse_coo_tensor(
            indices, values, torch.Size(adj_hat.shape)
        ).to(self.device)
 
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, num_model=None, **kwargs):
        if not "edge_index" in kwargs:
            raise ValueError("edge index not found in kwargs")
        
        if not num_model is None:
            return self.model_list[num_model].forward(x, labels, **{"edge_index": kwargs["edge_index"]})

        outputs = self.model_list[0](x, labels)
        if labels is not None:
            loss = outputs[1]
            outputs = outputs[0]

        ft_nl = [outputs.detach()]
        
        if self.feed_hidden_layer:
            ft_nl = [self.model_list[0].hidden_features]
        
        for it in range(1, self.nl + 1):
            outputs = self.model_list[it](torch.cat(ft_nl, 1), labels, **{"edge_index": kwargs["edge_index"]})
            if labels is not None:
                outputs = outputs[0]
                loss = outputs[1]

            ft_nl = [outputs.detach()]

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
        eps: float = 0.0
        eps_f: float = 0.0
        hidden_size: int = 10
        num_hidden: int = 2
        dropout: float = 0.0
        nl: int = 2

    model = MultiMLPGCN(RunConfig(), 10, 2, sparse.csr_matrix(torch.rand(10, 10)), "cpu", np.random)
    print(model)
    print(model.forward(torch.rand(10, 10)))