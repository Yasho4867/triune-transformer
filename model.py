import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
import warnings

# ─── FP4 via Transformer Engine ─────────────────────────────
try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    te = None

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if HAS_TE:
            self.linear = te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)
            
    def forward(self, x):
        return self.linear(x)
        
# ─── GLA ──────────────────────────────────────────────────────
try:
    from fla.ops.gla import chunk_gla
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    chunk_gla = None
    raise ImportError("flash-linear-attention is required. Install with: pip install flash-linear-attention")

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=ROPE_MAX_SEQ_LEN):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len, device):
        # Return in fp32 to preserve precision; caller casts q/k to bf16
        if seq_len <= self.max_seq_len:
            return self.cos_cached[:seq_len].to(device, torch.float32), self.sin_cached[:seq_len].to(device, torch.float32)

        # Generation can grow past the training sequence length.  Do not silently
        # return a too-short cache; build the required positions on demand.
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(q, k, cos, sin):
    # q, k are bf16; cos, sin are fp32 – we cast inside
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

class VectorisedGLA(nn.Module):
    def __init__(self, dim, heads, head_dim=GLA_HEAD_DIM):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.hidden_dim = heads * head_dim
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.use_rope = USE_ROPE
        if self.use_rope:
            self.rope = RotaryEmbedding(head_dim, max_seq_len=ROPE_MAX_SEQ_LEN)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        H, HD = self.heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, HD)
        k = self.k_proj(x).view(B, T, H, HD)
        v = self.v_proj(x).view(B, T, H, HD)
        # chunk_gla expects logarithmic decay gates, not probabilities.
        g = F.logsigmoid(self.g_proj(x).view(B, T, H, HD))

        # ELU+1 feature map – applied before RoPE (per GLA best practices)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        if self.use_rope:
            cos, sin = self.rope(T, x.device)
            # Apply RoPE after feature map – keeps relative position in the inner product
            q = q.transpose(1, 2)  # (B, H, T, HD)
            k = k.transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        # Use Triton kernel – required
        # FLA 0.5.1 uses batch/sequence/head layout and returns (output, state).
        out, _ = chunk_gla(q, k, v, g, scale=None)
        out = out.reshape(B, T, self.hidden_dim)
        return self.out_proj(out), None

# ─── RMSNorm ────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# ─── Hybrid Attention (only GLA, no dead projections) ────────
class HybridAttention(nn.Module):
    def __init__(self, dim, heads, use_fp4=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        # Only GLA – no separate q/k/v projections; they live inside GLA.
        self.gla = VectorisedGLA(dim, heads, GLA_HEAD_DIM)

    def forward(self, x, cache=None):
        return self.gla(x, cache)

# ─── MoE FFN ────────────────────────────────────────────────────
class MoE_FFN(nn.Module):
    def __init__(self, dim, num_experts=NUM_EXPERTS, top_k=TOP_K_EXPERTS, use_fp4=True, shared_expert=SHARED_EXPERT):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        self.shared_expert = shared_expert
        self.shared_scale = SHARED_EXPERT_SCALE
        LinearCls = FP4Linear if use_fp4 else nn.Linear

        self.experts = nn.ModuleList([
            nn.Sequential(
                LinearCls(dim, dim * EXPERT_HIDDEN_MULTIPLIER),
                nn.GELU(),
                LinearCls(dim * EXPERT_HIDDEN_MULTIPLIER, dim)
            )
            for _ in range(num_experts)
        ])

        if shared_expert:
            self.shared = nn.Sequential(
                LinearCls(dim, dim * EXPERT_HIDDEN_MULTIPLIER),
                nn.GELU(),
                LinearCls(dim * EXPERT_HIDDEN_MULTIPLIER, dim)
            )

        self.router = nn.Linear(dim, num_experts)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.last_centroids = None
        self.overflow_counter = 0
        self._global_step = 0

    def forward(self, x, update_stats=True):
        """If update_stats=False, skip bias/centroid updates (used during label generation)."""
        B, T, D = x.shape
        flat_x = x.reshape(B * T, D)

        bias_free_logits = self.router(x)
        biased_logits = bias_free_logits + self.expert_bias

        if self.training and update_stats:
            temp = max(0.1, 1.0 - self._global_step / 5000)
            gate_weights = F.softmax(bias_free_logits / temp, dim=-1)
        else:
            gate_weights = F.softmax(bias_free_logits, dim=-1)

        top_vals, top_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        flat_idx = top_idx.reshape(B * T, self.top_k)
        flat_vals = torch.gather(gate_weights, dim=-1, index=top_idx).reshape(B * T, self.top_k)

        capacity = int(math.ceil((B*T) / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        out = torch.zeros_like(flat_x)

        if self.shared_expert:
            shared_out = self.shared(flat_x)
            out += self.shared_scale * shared_out

        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            val_k = flat_vals[:, k]
            for e in range(self.num_experts):
                mask = (idx_k == e)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                dropped = indices.numel() - capacity
                if dropped > 0:
                    scores = val_k[indices]
                    keep = torch.argsort(scores, descending=True)[:capacity]
                    self.overflow_counter += dropped
                    indices = indices[keep]
                    val_k_keep = val_k[indices]
                else:
                    val_k_keep = val_k[indices]

                expert_input = flat_x[indices]

                orig_tokens = expert_input.size(0)
                pad = (-orig_tokens) % 16

                if pad:
                     expert_input = F.pad(expert_input, (0, 0, 0, pad))

                expert_output = self.experts[e](expert_input)

                if pad:
                    expert_output = expert_output[:orig_tokens]

                out[indices] += expert_output * val_k_keep.unsqueeze(-1)

        if self.training and update_stats:
            self._update_routing_stats(flat_x, flat_idx)

        return out.reshape(B, T, D)

    @torch.no_grad()
    def _update_routing_stats(self, flat_x, flat_idx):
        """Update non-gradient routing state without retaining an activation graph."""
        counts = torch.bincount(flat_idx.flatten(), minlength=self.num_experts)
        target = (flat_x.size(0) * self.top_k) / self.num_experts
        self.expert_bias += (counts - target).sign() * MOE_BIAS_UPDATE_RATE
        self.expert_bias.clamp_(-5.0, 5.0)

        centroids = []
        for expert_idx in range(self.num_experts):
            indices = (flat_idx == expert_idx).any(dim=1).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                centroids.append(flat_x[indices].mean(dim=0))
            else:
                centroids.append(torch.zeros(self.dim, device=flat_x.device, dtype=flat_x.dtype))
        self.last_centroids = torch.stack(centroids)

    @torch.no_grad()
    def update_routing_stats(self, x):
        """Update routing state for checkpointed forwards exactly once."""
        B, T, D = x.shape
        flat_x = x.detach().reshape(B * T, D)
        top_idx = torch.topk(self.router(x) + self.expert_bias, self.top_k, dim=-1).indices
        self._update_routing_stats(flat_x, top_idx.reshape(B * T, self.top_k))

# ─── Transformer Block ──────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, layer_idx, vocab_size, exit_layers, use_moe=True, use_fp4=True):
        super().__init__()
        self.attn = HybridAttention(dim, heads, use_fp4=use_fp4)
        self.ffn = MoE_FFN(dim, use_fp4=use_fp4) if use_moe else nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)
        )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.layer_idx = layer_idx
        if layer_idx in exit_layers:
            self.exit_head = nn.Linear(dim, vocab_size)
        else:
            self.exit_head = None
        self.use_moe = use_moe
        self._use_gradient_checkpointing = False

    def forward(self, x, return_exit=False, cache=None, update_stats=True):
        if self._use_gradient_checkpointing and self.training:
            attn_out, new_cache = torch.utils.checkpoint.checkpoint(
                self.attn, self.norm1(x), use_reentrant=False
            )
            x = x + attn_out
            norm2_x = self.norm2(x)
            if isinstance(self.ffn, MoE_FFN):
                ffn_out = torch.utils.checkpoint.checkpoint(
                    lambda hidden: self.ffn(hidden, update_stats=False),
                    norm2_x, use_reentrant=False
                )
                if update_stats:
                    self.ffn.update_routing_stats(norm2_x)
            else:
                ffn_out = torch.utils.checkpoint.checkpoint(
                    self.ffn, norm2_x, use_reentrant=False
                )
            x = x + ffn_out
        else:
            attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
            x = x + attn_out
            if isinstance(self.ffn, MoE_FFN):
                x = x + self.ffn(self.norm2(x), update_stats=update_stats)
            else:
                x = x + self.ffn(self.norm2(x))
        if self.exit_head and return_exit:
            exit_logits = self.exit_head(x)
            return x, exit_logits, new_cache
        return x, None, new_cache

# ─── TriuneTransformer ──────────────────────────────────────────
class TriuneTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, use_fp4=True):
        super().__init__()
        if num_layers <= LIMBIC_EXIT_LAYER:
            raise ValueError(
                f"num_layers must exceed LIMBIC_EXIT_LAYER ({LIMBIC_EXIT_LAYER}); got {num_layers}"
            )
        expected_hidden_dim = NUM_HEADS * GLA_HEAD_DIM
        if hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal NUM_HEADS * GLA_HEAD_DIM ({expected_hidden_dim})"
            )
        self.num_layers = num_layers
        self.router_prefix_layers = ROUTER_PREFIX_LAYERS
        self.reflex_exit_layer = REFLEX_EXIT_LAYER
        self.limbic_exit_layer = LIMBIC_EXIT_LAYER
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                NUM_HEADS,
                i,
                vocab_size,
                (self.reflex_exit_layer, self.limbic_exit_layer),
                use_moe=(i > self.reflex_exit_layer),
                use_fp4=use_fp4,
            )
            for i in range(num_layers)
        ])
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.final_head = nn.Linear(hidden_dim, vocab_size)
        self._use_gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and isinstance(m, (nn.Linear, getattr(te, 'Linear', nn.Linear))):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def gradient_checkpointing_enable(self):
        self._use_gradient_checkpointing = True
        for layer in self.layers:
            layer._use_gradient_checkpointing = True

    def _forward_block(self, layer, x, return_exit, cache=None, update_stats=True):
        return layer(x, return_exit, cache, update_stats=update_stats)

    def _run_layers(self, x, start_layer, end_layer, update_stats=True):
        for i in range(start_layer, end_layer):
            x, _, _ = self._forward_block(self.layers[i], x, False, update_stats=update_stats)
        return x

    def _route(self, x, force_depth, B, device):
        pooled = x.mean(dim=1)
        route_logits = self.router(pooled)
        if force_depth is None:
            depth_choice = route_logits.argmax(dim=-1)
        else:
            depth_choice = torch.full((B,), force_depth, device=device, dtype=torch.long)
        return route_logits, depth_choice

    def forward(self, input_ids, force_depth=None, cache=None):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x_prefix = self._run_layers(x, 0, self.router_prefix_layers)
        route_logits, depth_choice = self._route(x_prefix, force_depth, B, device)

        if force_depth is not None:
            if force_depth not in (0, 1, 2):
                raise ValueError(f"force_depth must be 0, 1, or 2; got {force_depth}")
            if force_depth == 0:
                x_out = self._run_layers(x_prefix, self.router_prefix_layers, self.reflex_exit_layer)
                _, logits, _ = self._forward_block(self.layers[self.reflex_exit_layer], x_out, True)
            elif force_depth == 1:
                x_out = self._run_layers(x_prefix, self.router_prefix_layers, self.limbic_exit_layer)
                _, logits, _ = self._forward_block(self.layers[self.limbic_exit_layer], x_out, True)
            else:
                x_out = self._run_layers(x_prefix, self.router_prefix_layers, self.num_layers)
                logits = self.final_head(self.final_norm(x_out))
            return logits, route_logits

        final_logits = torch.empty(B, T, self.final_head.out_features, device=device, dtype=x.dtype)
        for d in (0, 1, 2):
            idx = (depth_choice == d).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_d = x_prefix.index_select(0, idx)
            if d == 0:
                x_d = self._run_layers(x_d, self.router_prefix_layers, self.reflex_exit_layer)
                _, logits_d, _ = self._forward_block(self.layers[self.reflex_exit_layer], x_d, True)
            elif d == 1:
                x_d = self._run_layers(x_d, self.router_prefix_layers, self.limbic_exit_layer)
                _, logits_d, _ = self._forward_block(self.layers[self.limbic_exit_layer], x_d, True)
            else:
                x_d = self._run_layers(x_d, self.router_prefix_layers, self.num_layers)
                logits_d = self.final_head(self.final_norm(x_d))
            final_logits.index_copy_(0, idx, logits_d)
        return final_logits, route_logits

    def forward_all_exits(self, input_ids, update_stats=False):
        """Generate labels for depth router. If update_stats=False, MoE layers skip bias/centroid updates."""
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x_prefix = self._run_layers(x, 0, self.router_prefix_layers)
        route_logits, _ = self._route(x_prefix, None, B, device)

        # Run every MoE block with the same update_stats value.  Label generation
        # must not alter routing statistics.
        x6 = self._run_layers(
            x_prefix, self.router_prefix_layers, self.reflex_exit_layer, update_stats=update_stats
        )
        reflex_out, reflex_logits, _ = self._forward_block(
            self.layers[self.reflex_exit_layer], x6, True, update_stats=update_stats
        )

        # Limbic: continue from reflex_out (layers 7-15) then layer 16 full block
        x16 = self._run_layers(
            reflex_out, self.reflex_exit_layer + 1, self.limbic_exit_layer, update_stats=update_stats
        )
        limbic_out, limbic_logits, _ = self._forward_block(
            self.layers[self.limbic_exit_layer], x16, True, update_stats=update_stats
        )

        # Cortex: continue from limbic_out (layers 17-23)
        x24 = self._run_layers(
            limbic_out, self.limbic_exit_layer + 1, self.num_layers, update_stats=update_stats
        )
        cortex_logits = self.final_head(self.final_norm(x24))

        return reflex_logits, limbic_logits, cortex_logits, route_logits