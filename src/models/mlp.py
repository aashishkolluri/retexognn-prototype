from models.general_models import *

class MLP(GeneralModel):
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

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        for i in range(self.num_hidden-1):
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear_layers_list[self.num_hidden-1](x)
        if labels is None:
            return x
        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss

    def get_hidden_features(self):
        def hook(model, input, output):
            self.hidden_features = output.detach()
        return hook
    
    def register_forward_hook_for_hidden_features(self, layer=-2):
        self.linear_layers_list[layer].register_forward_hook(self.get_hidden_features())

if __name__ == "__main__":
    model = MLP(10, 8, 2)
    print(model)
    print(model.forward(torch.rand(10, 10)))