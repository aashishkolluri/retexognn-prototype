from models.general_models import * 
from models.dglmlp import DGLMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np

class DGLMLPPool(DGLMLP):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        pool_size=512,
        dropout=0.0,
        model_name="dgl_mlp_pool",
        num_hidden=2,
        rng=np.random
    ):
        super(DGLMLPPool, self).__init__(input_size+pool_size, hidden_size, output_size, dropout, num_hidden, model_name)

        self.pool_linear_layer = nn.Linear(input_size, pool_size, bias=False)
        self.num_hidden = num_hidden
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.pool_linear_layer.weight, gain=gain)
        
    def forward(self, mfgs, feat: torch.Tensor, **kwargs):
        mfg = mfgs[0]
        with mfg.local_scope():
            feat_src = feat
            feat_dst = feat[:mfgs[0].num_dst_nodes()]
            
            msg_fn = fn.copy_u('h', 'm')
            x = self.pool_linear_layer(feat_src)
            mfg.srcdata['h'] = x
            
            mfg.update_all(message_func=msg_fn, reduce_func=fn.max('m', 'h_N'))
            h_N = mfg.dstdata['h_N']
            
            x = torch.cat((feat_dst, h_N), dim=1)
        
            for i in range(self.num_hidden-1):
                # x = x[:mfgs[i].num_dst_nodes()]
                x = self.linear_layers_list[i](x)
                x = self.relu(x)
                x = self.dropout(x)
                
            # x = x[:mfgs[-1].num_dst_nodes()]
            x = self.linear_layers_list[self.num_hidden-1](x)
            
            return x
        
class MultiMLPDGLSage(GeneralMultiMLPModel):
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
        model_name="mmlpdgl_sage",
    ):
        super(MultiMLPDGLSage, self).__init__(run_config, input_size, output_size, device, rng, eps, num_hidden, model_name)
        
        self.model_name = model_name + "_nl_" + str(self.nl)
        self.input_size = input_size
        self.output_size = output_size
        self.feed_hidden_layer = feed_hidden_layer
        
        model = DGLMLP(
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
            
        for it in range(1, run_config.nl + 1):
            model = DGLMLPPool(
                model_name=f"{model_name}_dglmlpsage_{self.nl}_{it}",
                input_size=input_size,
                hidden_size=self.hidden_size,
                output_size=output_size,
                dropout=self.dropout_val,
                num_hidden=num_hidden
            )
            
            self.model_list.append(model)
            input_size = output_size
    
    def forward(self, mfgs, x: torch.Tensor, **kwargs):
        outputs = self.model_list[0](mfgs, x, **kwargs)
        
        ft_nl = [outputs.detach()]
        
        if self.feed_hidden_layer:
            ft_nl = [self.model_list[0].hidden_features]
            
        for it in range(1, self.nl + 1):
            ft_nl = torch.cat(ft_nl, 1)
            outputs = self.model_list[it](mfgs, ft_nl, **kwargs)
            
            ft_nl = [outputs.detach()]
            
        return outputs
            