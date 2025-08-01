import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # Simpler shape: [1, seq_len, 1, dim]
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(qkv, cos, sin):
    # Split qkv into q, k, v
    q, k, v = qkv.unbind(dim=2)  # [batch, seq_len, head, dim]
    
    # Apply rotary embeddings to q and k only
    def rotate_tokens(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    # Expand cos and sin to match q/k dimensions
    cos = cos.expand(q.shape[0], -1, q.shape[2], -1)  # [batch, seq_len, head, dim]
    sin = sin.expand(q.shape[0], -1, q.shape[2], -1)
    
    # Apply rotation
    q_out = (q * cos) + (rotate_tokens(q) * sin)
    k_out = (k * cos) + (rotate_tokens(k) * sin)
    
    # Stack back together with v (which remains unchanged)
    return torch.stack((q_out, k_out, v), dim=2)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockWithCrossAttention(nn.Module):
    def __init__(self, dim, n_heads, cond_dim=768, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.text_dim = cond_dim
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        
        # Self attention components
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # Cross attention components
        self.norm_cross = LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)
        self.dropout_cross = nn.Dropout(dropout)

        # MLP components
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)

        # Conditioning components
        self.adaLN_modulation = nn.Linear(cond_dim, 8 * dim, bias=True)  # 8 = 2 * 4 modulation signals
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, text_embeddings, text_attention_mask=None):
        # # Add dimension checks
        # print(f"Input shapes:")
        # print(f"x: {x.shape}")
        # print(f"text_embeddings: {text_embeddings.shape}")
        
        # if len(text_embeddings.shape) != 3 or text_embeddings.shape[-1] != self.text_dim:
        #     raise ValueError(
        #         f"Expected text_embeddings shape [batch, seq_len, {self.text_dim}], "
        #         f"got {text_embeddings.shape}"
        #     )

        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Split modulation signals
        (shift_msa, scale_msa,
         shift_cross, scale_cross,
         shift_mlp, scale_mlp,
         gate_msa, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(8, dim=2)

        # Self attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', 
                        three=3, h=self.n_heads, 
                        d=self.head_dim)
        
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
        # Replace flash attention with regular attention
        qkv = rearrange(qkv, 'b s three h d -> b h s (three d)')
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        
        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.attn_out(x)

        # First residual connection
        x = bias_dropout_scale_fn(x, None, gate_msa, x_skip, self.dropout1.p)

        # Cross attention
        x_skip = x
        x = modulate_fused(self.norm_cross(x), shift_cross, scale_cross)
        
        # Add missing sequence dimension to text_embeddings if needed
        if len(text_embeddings.shape) == 2:
            text_embeddings = text_embeddings.unsqueeze(1)  # [batch, 1, dim]
        
        # Project query from model features
        q = self.cross_q(x)  # [batch, seq_len, dim]
        
        # Project key and value from text embeddings (already in model dim)
        kv = self.cross_kv(text_embeddings)  # [batch, text_seq_len, 2*dim]
        
        k, v = kv.chunk(2, dim=-1)  # Each is [batch, text_seq_len, dim]
        
        # Reshape for attention
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)

        # Scale query for numerical stability
        q = q * (self.head_dim ** -0.5)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # # Apply attention mask if provided
        # if text_attention_mask is not None:
        #     attn = attn.masked_fill(
        #         ~text_attention_mask.unsqueeze(1).unsqueeze(2),
        #         float('-inf')
        #     )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_cross(attn)
        
        # Apply attention to values
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.cross_out(x)
        
        # Second residual connection
        x = x + x_skip

        # MLP block
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout2.p)

        return x

class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      
      # Replace flash attention with regular attention
      qkv = rearrange(qkv, 'b s three h d -> b h s (three d)')
      q, k, v = qkv.chunk(3, dim=-1)

      # Compute attention scores
      attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
      attn = F.softmax(attn, dim=-1)
      x = torch.matmul(attn, v)
      
      x = rearrange(x, 'b h s d -> b s (h d)')
      
    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x

class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, text_embed_dim=512):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        # Standard DiT components
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)
        
        # Text embedding projection
        self.text_proj = nn.Linear(text_embed_dim, 768)  # 768 is BERT hidden size
        print(f"Hidden Size: {config.model.hidden_size}")
        
        # Position embeddings for input sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, config.model.length, config.model.hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks with cross attention
        self.blocks = nn.ModuleList([
            DDiTBlockWithCrossAttention(
                dim=config.model.hidden_size,
                n_heads=config.model.n_heads,
                cond_dim=config.model.cond_dim,
                dropout=config.model.dropout
            ) for _ in range(config.model.n_blocks)
        ])

        # Output layer
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size,
            vocab_size,
            config.model.cond_dim
        )
        
        self.scale_by_sigma = config.model.scale_by_sigma
    def forward(self, indices, sigma, text_embeddings=None, text_attention_mask=None):
        """
        Args:
            indices: Input token indices [batch_size, seq_len]
            sigma: Noise level [batch_size]
            text_embeddings: BERT text embeddings [batch_size, text_seq_len, 768]
            text_attention_mask: Attention mask for text [batch_size, text_seq_len]
        """
        # Input embeddings
        # print(self.vocab_embed.embedding[0])
        x = self.vocab_embed(indices)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Time conditioning
        c = F.silu(self.sigma_map(sigma))
        
        # Process text embeddings if provided
        if text_embeddings is not None:
            # Project text embeddings to model dimension
            text_proj = self.text_proj(text_embeddings)
        else:
            # Use dummy text embeddings if none provided
            text_proj = torch.zeros(
                (x.shape[0], 1, x.shape[-1]), 
                device=x.device, 
                dtype=x.dtype
            )
            
        text_attention_mask = torch.ones(
            (x.shape[0], 1), 
            device=x.device, 
            dtype=torch.bool
        )

        # Get rotary embeddings
        rotary_cos_sin = self.rotary_emb(x)

        # Process through transformer blocks
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(
                    x, 
                    rotary_cos_sin, 
                    c, 
                    text_proj, 
                    text_attention_mask
                )
            x = self.output_layer(x, c)
            # print("Debug x after output layer:", x[0,0,:5])
            if x.isnan().any():
                print("NaN detected in x")
                # print("self.vocab_size:", self.vocab_size)
                # print('max index:', torch.max(indices))
                # print("indices:", indices)
                exit()
        
        return x