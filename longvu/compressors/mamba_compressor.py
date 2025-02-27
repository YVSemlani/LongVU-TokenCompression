import torch
from einops import rearrange
from torch import nn

from mamba_ssm import Mamba


# PUT THE BIMAMBAS BACK


class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        expand=2,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand

        dim = d_model * expand
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.in_proj = nn.Linear(d_model, dim, bias=True)

        self.out_proj = nn.Linear(dim, d_model, bias=True)

    def forward(self, x):
        x = self.in_proj(x)

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.out_proj(x)
        return x


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        layer_idx,
        use_norm=True,
        use_res=True,
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_norm = use_norm
        self.use_res = use_res
        if use_norm:
            self.norm = MambaRMSNorm(d_model)

        self.mixer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            #bimamba=bimamba,
        )
        # self.mixer = Attention(
        #     d_model=d_model,
        # )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        residual = hidden_states
        if self.use_norm:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        if self.use_res:
            hidden_states = residual + hidden_states
        return hidden_states


class MambaCompressorMultiScale(nn.Module):
    def __init__(
        self,
        d_model,
        n_layer,
        use_norm=True,
        use_res=True,
        fp32=True,
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
        pooling_method="avg2d",
    ):
        super().__init__()
        self.fp32 = fp32
        self.pooling_method = pooling_method
        # Use pooling for multi-scale
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    idx,
                    use_norm=use_norm,
                    use_res=use_res,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    #bimamba=bimamba,
                )
                for idx in range(n_layer)
            ]
        )
        if fp32:
            self.layers.to(torch.float32)

        if pooling_method == "avg2d":
            self.pool = nn.AvgPool2d(2)
        elif pooling_method == "avg3d":
            self.pool = nn.AvgPool3d(2)

    def forward(self, hidden_states):
        for mixer_block in self.layers:
            b, f, h, w, c = hidden_states.shape
            if self.fp32:
                dtype_prev = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
            hidden_states = hidden_states.reshape(b, -1, c)

            hidden_states = mixer_block(hidden_states)

            hidden_states = hidden_states.reshape(b, f, h, w, c)
            if self.fp32:
                hidden_states = hidden_states.to(dtype_prev)

            if self.pooling_method == "avg2d":
                hidden_states = rearrange(hidden_states, "b f h w c -> b (f c) h w")
                hidden_states = self.pool(hidden_states)
                hidden_states = rearrange(
                    hidden_states, "b (f c) h w -> b f h w c", f=f
                )
            elif self.pooling_method == "avg3d":
                hidden_states = rearrange(hidden_states, "b f h w c -> b c f h w")
                hidden_states = self.pool(hidden_states)
                hidden_states = rearrange(hidden_states, "b c f h w -> b f h w c")

        return hidden_states


class MambaCompressorQuery(nn.Module):
    def __init__(
        self,
        d_model,
        n_layer=1,
        use_norm=True,
        use_res=True,
        bf16=True,
        query_pos="inter",
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
        multi_scale=True,
        question_condition=False,
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.bf16 = bf16
        self.query_pos = query_pos
        self.question_condition = question_condition
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    idx,
                    use_norm=use_norm,
                    use_res=use_res,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    #bimamba=bimamba,
                )
                for idx in range(n_layer)
            ]
        )

        if bf16:
            self.layers.to(torch.bfloat16)

    def forward(self, space_time_tokens, hidden_states, question_states=None):
        # space_time_tokens is video features w/ dimensins (# frames, # of tokens per frame, hidden dim.)
        # we unsqueeze it to (# batch size, # frames, # of tokens per frame, hidden dim.)

        # hidden_states is presumably your learnable query tokens of shape (# of learnable query tokens, hidden dim.)

        # adding batch dimension to space_time_tokens and hidden_states
        space_time_tokens = space_time_tokens.unsqueeze(0)
        hidden_states = hidden_states.unsqueeze(0)

        # getting batch size, # frames, # of tokens per frame, hidden dim.
        b, f, hw, c = space_time_tokens.shape
        n_query = hidden_states.shape[1]

        for mixer_block in self.layers:
            space_time_tokens = space_time_tokens.reshape(b, -1, c)
            if self.question_condition:
                space_time_tokens = torch.cat(
                    (question_states, space_time_tokens), dim=1
                )

            if self.query_pos == "right":
                combined_tokens = torch.cat((space_time_tokens, hidden_states), dim=1)
            elif self.query_pos == "inter":
                combined_tokens = torch.zeros(
                    space_time_tokens.shape[0],
                    space_time_tokens.shape[1] + hidden_states.shape[1],
                    space_time_tokens.shape[2],
                ).to(hidden_states.device, dtype=hidden_states.dtype)
                mask = torch.zeros(combined_tokens.shape[1], dtype=bool)
                indices = torch.linspace(
                    0,
                    combined_tokens.shape[1] - 1,
                    hidden_states.shape[1] + 1,
                    dtype=int,
                )[1:]
                mask[indices] = True
                combined_tokens[:, mask] = hidden_states
                combined_tokens[:, ~mask] = space_time_tokens
            
            if self.bf16:
                dtype_prev = combined_tokens.dtype
                combined_tokens = combined_tokens.to(torch.bfloat16)
            combined_tokens = mixer_block(combined_tokens)
            if self.bf16:
                combined_tokens = combined_tokens.to(dtype_prev)

            if self.query_pos == "right":
                hidden_states = combined_tokens[:, -n_query:, :]
                space_time_tokens = combined_tokens[:, :-n_query, :]
            elif self.query_pos == "inter":
                hidden_states = combined_tokens[:, mask]
                space_time_tokens = combined_tokens[:, ~mask]

            if self.multi_scale:
                space_time_tokens = space_time_tokens.reshape(b, -1, hw, c)
                space_time_tokens = space_time_tokens[:, ::2]

        return hidden_states