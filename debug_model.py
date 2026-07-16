import torch
from config import *
from model import TriuneTransformer

# ─── Print config values ──────────────────────────────────────
print("=" * 60)
print("CONFIG VALUES:")
print(f"VOCAB_SIZE   = {VOCAB_SIZE}")
print(f"HIDDEN_DIM   = {HIDDEN_DIM}")
print(f"NUM_LAYERS   = {NUM_LAYERS}")
print(f"NUM_HEADS    = {NUM_HEADS}")
print(f"GLA_HEAD_DIM = {GLA_HEAD_DIM}")
print("=" * 60)

# ─── Instantiate model ────────────────────────────────────────
model = TriuneTransformer(
    vocab_size=VOCAB_SIZE,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

# ─── Inspect actual dimensions ───────────────────────────────
print("\nMODEL INSPECTION:")
print(f"Number of layers in model: {len(model.layers)}")

# Check first layer
first_layer = model.layers[0]
print(f"\nLayer 0 type: {type(first_layer).__name__}")
if hasattr(first_layer, 'attn'):
    attn = first_layer.attn
    gla = attn.gla
    print(f"  - attn.gla.q_proj.in_features: {gla.q_proj.in_features}")
    print(f"  - attn.gla.k_proj.in_features: {gla.k_proj.in_features}")
    print(f"  - attn.gla.v_proj.in_features: {gla.v_proj.in_features}")
    print(f"  - attn.gla.out_proj.in_features: {gla.out_proj.in_features}")
    print(f"  - attn.gla.heads: {gla.heads}")

# Check last layer (layer 23)
last_layer = model.layers[-1]
print(f"\nLayer {len(model.layers)-1} type: {type(last_layer).__name__}")
if hasattr(last_layer, 'attn'):
    attn = last_layer.attn
    gla = attn.gla
    print(f"  - attn.gla.q_proj.in_features: {gla.q_proj.in_features}")
    print(f"  - attn.gla.k_proj.in_features: {gla.k_proj.in_features}")
    print(f"  - attn.gla.v_proj.in_features: {gla.v_proj.in_features}")
    print(f"  - attn.gla.out_proj.in_features: {gla.out_proj.in_features}")
    print(f"  - attn.gla.heads: {gla.heads}")

# Check MoE FFN in a layer > 6 (e.g., layer 7)
if len(model.layers) > 7:
    moe_layer = model.layers[7]
    if hasattr(moe_layer, 'ffn') and hasattr(moe_layer.ffn, 'experts'):
        print(f"\nLayer 7 MoE FFN:")
        first = getattr(moe_layer.ffn.experts[0][0], "linear", moe_layer.ffn.experts[0][0])
        second = getattr(moe_layer.ffn.experts[0][2], "linear", moe_layer.ffn.experts[0][2])
        print(f"  - ffn.experts[0][0].in_features: {first.in_features}")
        print(f"  - ffn.experts[0][0].out_features: {first.out_features}")
        print(f"  - ffn.experts[0][2].in_features: {second.in_features}")
        print(f"  - ffn.experts[0][2].out_features: {second.out_features}")

# ─── Total parameter count ────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ─── Theoretical parameter count based on config ─────────────
# Rough estimate:
# Each TransformerBlock has:
#   - attn: q,k,v,out: 4 * hidden_dim^2
#   - ffn (MoE): for each expert: 2 * hidden_dim * 4*hidden_dim = 8*hidden_dim^2 per expert?
#   Actually MoE has 3 experts, each with two linear layers: (hidden_dim -> 4*hidden_dim) and (4*hidden_dim -> hidden_dim)
#   That's 2 * hidden_dim * 4*hidden_dim = 8*hidden_dim^2 per expert * 3 = 24*hidden_dim^2
#   Plus router: hidden_dim * num_experts
#   Plus norms, embeddings, etc.
# Rough theoretical:
if NUM_LAYERS > 0:
    # Attention params per layer: q,k,v,out = 4 * HIDDEN_DIM^2
    attn_params = 4 * HIDDEN_DIM * HIDDEN_DIM
    # MoE params per layer (for layers > 6): 3 experts * (HIDDEN_DIM * 4*HIDDEN_DIM + 4*HIDDEN_DIM * HIDDEN_DIM) = 3 * 8 * HIDDEN_DIM^2 = 24 * HIDDEN_DIM^2
    moe_params = 24 * HIDDEN_DIM * HIDDEN_DIM
    # router per layer: HIDDEN_DIM * NUM_EXPERTS
    router_params = HIDDEN_DIM * NUM_EXPERTS
    # Norms: 2 * HIDDEN_DIM per layer
    norm_params = 2 * HIDDEN_DIM
    # Embedding + final head: 2 * VOCAB_SIZE * HIDDEN_DIM
    embed_params = 2 * VOCAB_SIZE * HIDDEN_DIM
    layers_with_moe = NUM_LAYERS - 7  # layers 7..23 (since use_moe=(i>6))
    if layers_with_moe < 0:
        layers_with_moe = 0
    non_moe_layers = NUM_LAYERS - layers_with_moe
    total_theoretical = embed_params + non_moe_layers * (attn_params + norm_params) + layers_with_moe * (attn_params + moe_params + router_params + norm_params)
    print(f"\nTheoretical parameter estimate (approx): {total_theoretical:,}")

# ─── Check if config values are being overridden ─────────────
print("\n💡 If numbers don't match config, check for:")
print("   - Another config.py file being imported")
print("   - Environment variables overriding config")
print("   - The model class using different defaults")
