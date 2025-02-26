import torch
# from mamba_ssm import Mamba
from mamba_ssm import Mamba
from torch import nn

class TTTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class TTTRMSNorm(nn.Module):
    def __init__(self, d_model, layer_idx, use_norm = True, use_res = True, 
                 d_state=16, d_conv=4, expand=2, bimamba = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_norm = use_norm
        self.use_res = use_res
        if use_norm:
            self.norm = TTTRMSNorm(d_model)
        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, bimamba = bimamba)

    def forward(self, hidden_states):
        residual = hidden_states
        if self.use_norm:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        if self.use_res:
            hidden_states = residual + hidden_states
        return hidden_states
    
class TTTCompressor(nn.Module):
    def __init__(self, d_model, n_layer, use_norm=True, use_res = True, fp32 = True, 
                 query_pos = 'right', d_state=16, d_conv=4, expand=2, bimamba = True, multi_scale = True):
        super().__init__()
        self.multi_scale = multi_scale
        self.fp32 = fp32
        self.query_pos = query_pos
        self.layers = nn.ModuleList([
                TTTRMSNorm(d_model, idx, use_norm=use_norm, use_res=use_res, 
                           d_state=d_state, d_conv=d_conv, expand=expand, bimamba = bimamba) 
                for idx in range(n_layer)
                ])
                
        if fp32:
            self.layers.to(torch.float32) 
        
    def forward(self, space_time_tokens, hidden_states):
             
        b, f, h, w, c = space_time_tokens.shape
        n_query = hidden_states.shape[1]
        for mixer_block in self.layers:
            space_time_tokens = space_time_tokens.reshape(b, -1, c)
            if self.query_pos=='right':
                hidden_states = torch.cat((space_time_tokens, hidden_states), dim=1)
            elif self.query_pos=='inter':
                combined_tokens = torch.zeros(space_time_tokens.shape[0], space_time_tokens.shape[1]+hidden_states.shape[1], space_time_tokens.shape[2]).to(hidden_states.device, dtype = hidden_states.dtype)
                mask = torch.zeros(combined_tokens.shape[1], dtype=bool)
                indices = torch.linspace(0, combined_tokens.shape[1]-1, hidden_states.shape[1]+1, dtype=int)[1:]
                mask[indices] = True
                combined_tokens[:,mask] = hidden_states
                combined_tokens[:,~mask] = space_time_tokens
                # hidden_states = temp 
            
            if self.fp32:
                dtype_prev = combined_tokens.dtype
                combined_tokens = combined_tokens.to(torch.float32)  
            combined_tokens = mixer_block(combined_tokens)
            if self.fp32:
                combined_tokens = combined_tokens.to(dtype_prev)

            if self.query_pos=='right':
                hidden_states = combined_tokens[:,-n_query:,:]
                space_time_tokens = combined_tokens[:,:-n_query,:]
            elif self.query_pos=='inter':
                hidden_states = combined_tokens[:,mask]
                space_time_tokens = combined_tokens[:,~mask]
            
            if self.multi_scale:
                space_time_tokens = space_time_tokens.reshape(b, -1, h, w, c)
                space_time_tokens = space_time_tokens[:,::2]
            
        return hidden_states