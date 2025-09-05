import torch
from torch import nn, Tensor
from torch.amp import autocast
from typing import Optional
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
        Adds sinusoidal positional encoding to the input embeddings.
    """
    def __init__(self, d_model:int, max_len:int):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.init_pe()
    
    def init_pe(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2)/self.d_model*(-math.log(10000.0)))
        pe[:, 0::2] = torch.sin(position*div_term)
        if self.d_model%2 != 0: div_term = div_term[:-1] # For odd feature dim
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        if x.shape[1] <= self.max_len:
            x = x + self.pe[:x.shape[1]]
        else:
            raise ValueError(f"Input sequence length {x.shape[1]} exceeds maximum length {self.max_len}. Please increase max_len.")
        return x
    

# Adapted source code from: https://docs.pytorch.org/torchtune/0.2/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, *, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, n_h, s, h_d]
        seq_len = x.size(2)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, n_h, s, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, 1, s, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, 1, s, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, 1, seq_len, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dim_feedforward:int, dropout=0.1, relative_emb=None):
        super(MultiHeadedSelfAttention, self).__init__()
        self.dim = dim
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.head_dim = self.dim_feedforward // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, self.head_dim * num_heads)
        self.k_proj = nn.Linear(dim, self.head_dim * num_heads)
        self.v_proj = nn.Linear(dim, self.head_dim * num_heads)
        self.relative_emb = relative_emb
        
        self.sm = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
                
        # _zero_pad = torch.zeros(self.num_heads * self.head_dim - self.dim, dtype=torch.float32)
        # self.register_buffer('_zero_pad', _zero_pad)
        
    def build_max_history_mask(self, seq_len: int, max_history: int, device=None):
        idx = torch.arange(seq_len, device=device)
        # Compare broadcasted query and key indices
        mask = idx.unsqueeze(1) - idx.unsqueeze(0)  # (seq_len, seq_len)
        mask = mask > (max_history - 1)  # True where j < i - max_history + 1
        return mask.float().masked_fill(mask, float('-inf'))  # (seq_len, seq_len)
    
    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, is_causal:bool=False, key_padding_mask=None, max_history_len:int=None):
        """
        Scaled dot-product attention mechanism.
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
            attn_mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len). Use -inf for masked positions and 0 for unmasked positions.
            is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
            key_padding_mask (torch.Tensor, optional): Key padding mask of shape (batch_size, seq_len). Use 0 for unmasked positions and 1 for masked positions.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        bsz, num_heads, seq_len, head_dim = q.size()
        
        if self.relative_emb is not None and self.relative_emb.__class__ == RotaryPositionalEmbeddings:
            q, k = self.relative_emb(q), self.relative_emb(k)
        
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        
        # TODO: Add ALiBi relative positional bias from Press et al. (2021), https://arxiv.org/abs/2108.12409
        # if self.relative_emb is not None and self.relative_emb.__class__ == ALiBIPositionalBias:
        #     attn_weights = attn_weights + self.relative_emb(seq_len=q.size(2), num_heads=self.num_heads, device=q.device)
        
        if is_causal and attn_mask is None:
            # Create a causal mask if not provided
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).unsqueeze(0).unsqueeze(0)
            attn_mask = torch.where(attn_mask == 0, float('-inf'), 0.0).expand(bsz, 1, -1, -1)
        
        if max_history_len is not None:
            history_mask = self.build_max_history_mask(seq_len, max_history_len, device=q.device)
            history_mask = history_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            attn_mask = history_mask if attn_mask is None else attn_mask + history_mask
        
        if key_padding_mask is not None:
            # make the columns of attn_mask corresponding to pad tokens (key_padding_mask==True) equal to -inf
            # key_padding_mask is of shape (batch_size, seq_len)
            idxs = torch.nonzero(key_padding_mask==1)
            attn_mask = attn_mask.clone()
            attn_mask[idxs[:, 0], :, :, idxs[:, 1]] = float('-inf')
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn = self.sm(attn_weights)
        
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, -1)#[..., :self.dim]  # Remove padding if added
        
        # if return_attn: return out, attn        
        return out

    # @autocast(device_type='cuda')
    def forward(self, x, attn_mask=None, is_causal:bool=False, key_padding_mask=None, max_history_len:int=None):
        """
        Forward pass for the multi-headed self-attention mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            attn_mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len). Use -inf for masked positions and 0 for unmasked positions.
            is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
            key_padding_mask (torch.Tensor, optional): Key padding mask of shape (batch_size, seq_len). Use 0 for unmasked positions and 1 for masked positions.
            max_history_len (int, optional): Maximum history length for attention. If provided, applies a history mask.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        bsz, seq_len, h_dim = x.size()
        
        # Project input to query, key, value tensors. Shape: (batch_size, seq_len, 3 * dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # if h_dim % self.num_heads != 0:
        #     q = torch.cat([q, self._zero_pad.expand(bsz, seq_len, -1)], dim=-1)
        #     k = torch.cat([k, self._zero_pad.expand(bsz, seq_len, -1)], dim=-1)
        #     v = torch.cat([v, self._zero_pad.expand(bsz, seq_len, -1)], dim=-1)
                     
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Apply scaled dot-product attention
        out = self.scaled_dot_product_attention(q, k, v, 
                                                attn_mask=attn_mask, 
                                                is_causal=is_causal, 
                                                key_padding_mask=key_padding_mask,
                                                max_history_len=max_history_len)
        return out
    

# Classical Transformer block implementation from the paper "Attention is All You Need" (Vaswani et al., 2017)
# https://arxiv.org/abs/1706.03762
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward:int,
                 dropout:float=0.0, rotary_emb=None, activation:str='relu', pre_ln:bool=False,
                 max_history_len:int=None):
        super(TransformerBlock, self).__init__()
        self.dim = input_dim
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else input_dim
        self.num_heads = num_heads
        self.head_dim = self.dim_feedforward // self.num_heads
        self.max_history_len = max_history_len

        self.ln1 = nn.LayerNorm(self.dim, elementwise_affine=True)
        self.attn = MultiHeadedSelfAttention(dim=self.dim, num_heads=num_heads, dim_feedforward=self.dim_feedforward, 
                                             dropout=dropout, relative_emb=rotary_emb)
        self.out_proj = nn.Linear(self.num_heads*self.head_dim, self.dim)
        self.ln2 = nn.LayerNorm(self.dim, elementwise_affine=True)
        
        self.mlp_in = nn.Linear(self.dim, self.dim)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU() if activation == 'gelu' else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.mlp_out = nn.Linear(self.dim, self.dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_ln = pre_ln # Set to True for pre-layer normalization, False for post-layer normalization. Vaswani used post-layer normalization.
        self.init_model()
    
    def init_model(self):
        # init layers with xavier uniform initialization
        nn.init.xavier_uniform_(self.attn.q_proj.weight)
        nn.init.xavier_uniform_(self.attn.k_proj.weight)
        nn.init.xavier_uniform_(self.attn.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.mlp_in.weight)
        nn.init.xavier_uniform_(self.mlp_out.weight)
        nn.init.constant_(self.ln1.bias, 0.0)
        nn.init.constant_(self.ln2.bias, 0.0)
        nn.init.constant_(self.attn.q_proj.bias, 0.0)
        nn.init.constant_(self.attn.k_proj.bias, 0.0)
        nn.init.constant_(self.attn.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.constant_(self.mlp_in.bias, 0.0)
        nn.init.constant_(self.mlp_out.bias, 0.0)
        
    def forward(self, x, attn_mask=None, is_causal:bool=False, key_padding_mask=None):
        """
        Forward pass for the Transformer block with Rotary Position Embedding.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            attn_mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len). Use -inf for masked positions and 0 for unmasked positions.
            is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
            key_padding_mask (torch.Tensor, optional): Key padding mask of shape (batch_size, seq_len). Use 0 for unmasked positions and 1 for masked positions.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        if self.pre_ln:
            # Layer normalization 1
            x_ln = self.ln1(x)
            
            # Multi-headed self-attention
            attn_out = self.attn(x_ln, attn_mask=attn_mask,
                                    is_causal=is_causal, key_padding_mask=key_padding_mask,
                                    max_history_len=self.max_history_len)
            out = self.out_proj(attn_out) + x
            
            # Layer normalization 2
            z = self.dropout1(self.activation(self.mlp_in(self.ln2(out))))
            z = self.dropout2(self.mlp_out(z))
            z = z + out
        else:        
            # Multi-headed self-attention
            attn_out = self.attn(x, attn_mask=attn_mask,
                                    is_causal=is_causal, key_padding_mask=key_padding_mask,
                                    max_history_len=self.max_history_len)
            out = self.out_proj(attn_out) + x
            
            out_ln = self.ln1(out)
            
            # Layer normalization 2
            z = self.dropout1(self.activation(self.mlp_in(out_ln)))
            z = self.dropout2(self.mlp_out(z))
            z = self.ln2(z + out_ln)

        # TODO: make this code elegant. E.g. https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/nn/modules/transformer.py#L905
        return z


# Iimplementation of the parallel Transformer block: https://arxiv.org/abs/1911.09483 / https://github.com/kingoflolz/mesh-transformer-jax
# Implementation is based on depiction from https://arxiv.org/pdf/2311.01906
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, rotary_emb=None, activation:str='relu'):
        super(ParallelTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ln1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout, rotary_emb)
        self.out_proj = nn.Linear(dim, dim)
        
        self.mlp_in = nn.Linear(dim, dim)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU() if activation == 'gelu' else nn.Identity()
        self.mlp_out = nn.Linear(dim, dim)
        self.init_model()
        
    def init_model(self):
        # init layers with xavier uniform initialization
        nn.init.xavier_uniform_(self.mlp_in.weight)
        nn.init.xavier_uniform_(self.mlp_out.weight)
        nn.init.constant_(self.ln1.bias, 0.0)
        nn.init.constant_(self.mlp_in.bias, 0.0)
        nn.init.constant_(self.mlp_out.bias, 0.0)
        
    def forward(self, x, attn_mask=None, is_causal:bool=False, key_padding_mask=None):
        """
        Forward pass for the parallel Transformer block with Rotary Position Embedding.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            attn_mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len). Use -inf for masked positions and 0 for unmasked positions.
            is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
            key_padding_mask (torch.Tensor, optional): Key padding mask of shape (batch_size, seq_len). Use 0 for unmasked positions and 1 for masked positions.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """        
        # Layer normalization 1
        x = self.ln1(x)

        # Multi-headed self-attention
        attn_out = self.attn(x, attn_mask=attn_mask,
                                is_causal=is_causal, key_padding_mask=key_padding_mask)
        attn_out_proj = self.out_proj(attn_out)
        
        out = self.mlp_in(x)
        out = self.activation(out)
        out = self.mlp_out(out) 
        
        out = out + attn_out_proj
        out = out +  x
        return out


class RoPETransformerModel(nn.Module):
    def __init__(self, input_dim:int , num_heads:int , num_layers:int , num_classes:int, dim_feedforward:int, num_outcomes:int=1, pre_ln:bool=False, 
                 dropout:float=0.0, output_dropout:float=0.0, max_seq_len:int=512, max_history_len:int=None, mute_grad:bool=False):
        super(RoPETransformerModel, self).__init__()
        self.rotary_emb_obj = RotaryPositionalEmbeddings(dim=input_dim//num_heads, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(input_dim=input_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, rotary_emb=self.rotary_emb_obj, pre_ln=pre_ln, max_history_len=max_history_len) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(input_dim, num_classes * num_outcomes)
        self.op_dropout = nn.Dropout(output_dropout)
        self.max_seq_len = max_seq_len
        self.mute_grad = mute_grad

    def forward(self, embeddings, mask=None, **kwargs):
        # src_key_padding_mask only to mask out the padded tokens, i.e., timestep with time_ids = -1. 
        # Boolean src_key_padding_mask uses True to mask out the padded tokens. 
        key_padding_mask = torch.where(kwargs['time_ids']==-1, True, False)
        # Turn the mask into sq matrix of shape (batch_size, 1, seq_len, seq_len)
        # src_key_padding_mask_sq = src_key_padding_mask_float.unsqueeze(-1) @ src_key_padding_mask_float.unsqueeze(-1).transpose(-1, -2)
        # src_key_padding_mask_sq = torch.where(src_key_padding_mask_sq==0.0, float('-inf'), 0.0)
        
        # Generate a square mask for the transformer. This should be of size (seq_len, seq_len)
        # This mask is used to mask out the future tokens in the sequence.  
        # For float -inf is used to mask out the future tokens, 0s are used to keep the present and past tokens.
        causal_mask = ~torch.tril(torch.ones(embeddings.shape[1], embeddings.shape[1], device=embeddings.device)).bool()
        causal_mask = torch.where(causal_mask, float('-inf'), 0.0)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(embeddings.shape[0], -1, -1, -1)
 
        # src_key_padding_mask_sq = src_key_padding_mask_sq + causal_mask.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        # src_key_padding_mask_sq = src_key_padding_mask_sq.unsqueeze(1)  # Add a dimension for heads
        x = embeddings
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, 
                      is_causal=True, key_padding_mask=key_padding_mask)
        
        x = self.op_dropout(x)
        x = self.fc_out(x)
        return x
    
    
class SinusoidalTransformerModel(nn.Module):
    def __init__(self, input_dim:int , num_heads:int , num_layers:int , num_classes:int, dim_feedforward:int, num_outcomes:int=1, 
                 dropout:float=0.0, output_dropout:float=0.0, max_len:int=512, pre_ln:bool=False, max_history_len:int=None, 
                 mute_grad:bool=False):
        super(SinusoidalTransformerModel, self).__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(input_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(input_dim=input_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, pre_ln=pre_ln, max_history_len=max_history_len) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(input_dim, num_classes * num_outcomes)  
        self.op_dropout = nn.Dropout(output_dropout)
        self.max_len = max_len
        self.mute_grad = mute_grad

    def forward(self, embeddings, mask=None, **kwargs):
        
        # src_key_padding_mask only to mask out the padded tokens, i.e., timestep with time_ids = -1. 
        # Boolean src_key_padding_mask uses True to mask out the padded tokens. 
        key_padding_mask = torch.where(kwargs['time_ids']==-1, True, False)
        # Turn the mask into sq matrix of shape (batch_size, 1, seq_len, seq_len)
        # src_key_padding_mask_sq = src_key_padding_mask_float.unsqueeze(-1) @ src_key_padding_mask_float.unsqueeze(-1).transpose(-1, -2)
        # src_key_padding_mask_sq = torch.where(src_key_padding_mask_sq==0.0, float('-inf'), 0.0)
        
        # Generate a square mask for the transformer. This should be of size (seq_len, seq_len)
        # This mask is used to mask out the future tokens in the sequence.  
        # For float -inf is used to mask out the future tokens, 0s are used to keep the present and past tokens.
        causal_mask = ~torch.tril(torch.ones(embeddings.shape[1], embeddings.shape[1], device=embeddings.device)).bool()
        causal_mask = torch.where(causal_mask, float('-inf'), 0.0)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(embeddings.shape[0], -1, -1, -1)
 
        # src_key_padding_mask_sq = src_key_padding_mask_sq + causal_mask.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        # src_key_padding_mask_sq = src_key_padding_mask_sq.unsqueeze(1)  # Add a dimension for heads

        x = self.positional_encoding(embeddings)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, 
                      is_causal=True, key_padding_mask=key_padding_mask)
        
        x = self.op_dropout(x)
        y = self.fc_out(x)
        return y


class NoPositionalTransformerModel(nn.Module):
    def __init__(self, input_dim:int , num_heads:int , num_layers:int , num_classes:int, dim_feedforward:int, num_outcomes:int=1, pre_ln:bool=False, 
                 dropout:float=0.0, output_dropout:float=0.0, max_history_len:int=None, mute_grad:bool=False):
        super(NoPositionalTransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(input_dim=input_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, pre_ln=pre_ln, max_history_len=max_history_len) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(input_dim, num_classes * num_outcomes)
        self.op_dropout = nn.Dropout(output_dropout)
        self.mute_grad = mute_grad

    def forward(self, embeddings, mask=None, **kwargs):
        # src_key_padding_mask only to mask out the padded tokens, i.e., timestep with time_ids = -1. 
        # Boolean src_key_padding_mask uses True to mask out the padded tokens. 
        key_padding_mask = torch.where(kwargs['time_ids']==-1, True, False)
        # Turn the mask into sq matrix of shape (batch_size, 1, seq_len, seq_len)
        # src_key_padding_mask_sq = src_key_padding_mask_float.unsqueeze(-1) @ src_key_padding_mask_float.unsqueeze(-1).transpose(-1, -2)
        # src_key_padding_mask_sq = torch.where(src_key_padding_mask_sq==0.0, float('-inf'), 0.0)
        
        # Generate a square mask for the transformer. This should be of size (seq_len, seq_len)
        # This mask is used to mask out the future tokens in the sequence.  
        # For float -inf is used to mask out the future tokens, 0s are used to keep the present and past tokens.
        causal_mask = ~torch.tril(torch.ones(embeddings.shape[1], embeddings.shape[1], device=embeddings.device)).bool()
        causal_mask = torch.where(causal_mask, float('-inf'), 0.0)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(embeddings.shape[0], -1, -1, -1)
 
        # src_key_padding_mask_sq = src_key_padding_mask_sq + causal_mask.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        # src_key_padding_mask_sq = src_key_padding_mask_sq.unsqueeze(1)  # Add a dimension for heads
        x = embeddings
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, 
                      is_causal=True, key_padding_mask=key_padding_mask)
        
        x = self.op_dropout(x)
        y = self.fc_out(x)
        return y
    
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim:int , num_heads:int , num_layers:int , num_classes:int , dim_feedforward:int=None, num_outcomes:int=1,
                 dropout:float=0.0, output_dropout:float=0.0, max_len:int=512, 
                 positional_encoding:str='sinusoidal', pre_ln:bool=False, max_history_len:int=None, mute_grad:bool=False):
        super(TransformerModel, self).__init__()
        if positional_encoding == 'sinusoidal':
            self.model = SinusoidalTransformerModel(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward,
                                                    num_classes=num_classes, num_outcomes=num_outcomes, dropout=dropout, 
                                                    output_dropout=output_dropout, max_len=max_len, pre_ln=pre_ln, max_history_len=max_history_len)
        elif positional_encoding == 'rope':
            self.model = RoPETransformerModel(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward,
                                              num_classes=num_classes, num_outcomes=num_outcomes, dropout=dropout, 
                                              output_dropout=output_dropout, max_seq_len=max_len, pre_ln=pre_ln, max_history_len=max_history_len) 
        elif positional_encoding == 'none':
            self.model = NoPositionalTransformerModel(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward,
                                                      num_classes=num_classes, num_outcomes=num_outcomes, dropout=dropout, 
                                                      output_dropout=output_dropout, pre_ln=pre_ln, max_history_len=max_history_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding}. Choose from 'sinusoidal', 'rope', or 'none'.")
        self.mute_grad = mute_grad
        
    def forward(self, embeddings, mask=None, **kwargs):
        """
        Forward pass for the Transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len). Use -inf for masked positions and 0 for unmasked positions.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_classes).
        """
    
        return self.model(embeddings, mask, **kwargs)
    