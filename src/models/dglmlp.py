from models.general_models import *

class DGLMLP(GeneralModel):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_hidden=2,
        model_name="mlp",
    ):
        super().__init__(input_size, hidden_size, output_size, dropout, num_hidden, model_name)
        self.register_forward_hook_for_hidden_features()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        for layer in self.linear_layers_list:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
    
    def forward(self, mfgs, x: torch.Tensor, **kwargs):
        for i in range(self.num_hidden-1):
            x = x[:mfgs[i].num_dst_nodes()]
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = x[:mfgs[-1].num_dst_nodes()]
        x = self.linear_layers_list[self.num_hidden-1](x)
        return x

    def get_hidden_features(self):
        def hook(model, input, output):
            self.hidden_features = output.detach()
        return hook
    
    def register_forward_hook_for_hidden_features(self, layer=-2):
        self.linear_layers_list[layer].register_forward_hook(self.get_hidden_features())