import torch
import torch.nn as nn

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
        super(GeneralModel, self).__init__()

        self.model_name = model_name
        self.num_hidden = num_hidden
        self.linear_layers_list = nn.ModuleList()
        self.linear_layers_list.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, self.num_hidden-1):
            self.linear_layers_list.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers_list.append(nn.Linear(hidden_size, output_size))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        '''
        This function is to be defined in the child classes.
        '''
        pass