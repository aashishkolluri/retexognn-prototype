import torch
import torch.nn as nn
import os

class GeneralModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_hidden=2,
        model_name="general_model_to_be_inherited",
    ):
        super().__init__()

        self.model_name = model_name
        self.num_hidden = num_hidden
        self.linear_layers_list = nn.ModuleList()
        self.linear_layers_list.append(nn.Linear(input_size, hidden_size, bias=False))
        for _ in range(1, self.num_hidden-1):
            self.linear_layers_list.append(nn.Linear(hidden_size, hidden_size, bias=False))
        self.linear_layers_list.append(nn.Linear(hidden_size, output_size, bias=False))
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

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        '''
        This function is to be defined in the child classes.
        '''
        pass

class GeneralMultiMLPModel(nn.Module):
    def __init__(
        self,
        run_config,
        input_size,
        output_size,
        device,
        rng,
        eps=0.0,
        num_hidden=2,
        model_name="general_multi_mlp_model_to_be_inherited",
    ):
        super().__init__()

        self.input_size = input_size
        self.model_name = model_name
        self.output_size = output_size
        self.hidden_size = run_config.hidden_size
        self.num_hidden = num_hidden
        self.device = device
        self.dropout_val = run_config.dropout
        self.nl = run_config.nl
        self.eps = eps
        if eps > 0:
            self.dp = True
        else:
            self.dp = False
        self.rng = rng
        self.model_list = nn.ModuleList()

    def prepare_for_fwd(self):
        for model in self.model_list:
            model.eval()

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.model_list[0].linear_layers_list[0].weight.device # hacky way to get the device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        '''
        This function is to be defined in the child classes.
        '''
        pass
